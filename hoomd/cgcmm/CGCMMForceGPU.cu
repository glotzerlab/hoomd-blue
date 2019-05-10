// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "CGCMMForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file CGCMMForceGPU.cu
    \brief Defines GPU kernel code for calculating the Lennard-Jones pair forces. Used by CGCMMForceComputeGPU.
*/

//! Kernel for calculating CG-CMM Lennard-Jones forces
/*! This kernel is called to calculate the Lennard-Jones forces on all N particles for the CG-CMM model potential.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for reading \a d_nlist
    \param d_coeffs Coefficients to the lennard jones force (lj12, lj9, lj6, lj4).
    \param coeff_width Width of the coefficient matrix
    \param r_cutsq Precalculated r_cut*r_cut, where r_cut is the radius beyond which forces are
    set to 0
    \param box Box dimensions used to implement periodic boundary conditions

    \a coeffs is a pointer to a matrix in memory. \c coeffs[i*coeff_width+j].x is \a lj12 for the type pair \a i, \a j.
    Similarly, .y, .z, and .w are the \a lj9, \a lj6, and \a lj4 parameters, respectively. The values in d_coeffs are
    read into shared memory, so \c coeff_width*coeff_width*sizeof(Scalar4) bytes of extern shared memory must be allocated
    for the kernel call.

    Developer information:
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
__global__ void gpu_compute_cgcmm_forces_kernel(Scalar4* d_force,
                                                Scalar* d_virial,
                                                const unsigned int virial_pitch,
                                                const unsigned int N,
                                                const Scalar4 *d_pos,
                                                const BoxDim box,
                                                const unsigned int *d_n_neigh,
                                                const unsigned int *d_nlist,
                                                const unsigned int *d_head_list,
                                                const Scalar4 *d_coeffs,
                                                const int coeff_width,
                                                const Scalar r_cutsq)
    {
    // read in the coefficients
    extern __shared__ Scalar4 s_coeffs[];
    for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < coeff_width*coeff_width)
            s_coeffs[cur_offset + threadIdx.x] = d_coeffs[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];
    const unsigned int head_idx = d_head_list[idx];

    // read in the position of our particle.
    // (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // prefetch neighbor index
    unsigned int cur_neigh = 0;
    unsigned int next_neigh(0);
    next_neigh = __ldg(d_nlist + head_idx);

    // loop over neighbors
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
        {
        // read the current neighbor index (MEM TRANSFER: 4 bytes)
        // prefetch the next value and set the current one
        cur_neigh = next_neigh;
        next_neigh = __ldg(d_nlist + head_idx + neigh_idx+1);

        // get the neighbor's position (MEM TRANSFER: 16 bytes)
        Scalar4 neigh_postype = __ldg(d_pos + cur_neigh);
        Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

        // calculate dr (with periodic boundary conditions)
        Scalar3 dx = pos - neigh_pos;

        // apply periodic boundary conditions: (FLOPS 12)
        dx = box.minImage(dx);

        // calculate r squared (FLOPS: 5)
        Scalar rsq = dot(dx, dx);

        // calculate 1/r^2 (FLOPS: 2)
        Scalar r2inv;
        if (rsq >= r_cutsq)
            r2inv = Scalar(0.0);
        else
            r2inv = Scalar(1.0) / rsq;

        // lookup the coefficients between this combination of particle types
        int typ_pair = __scalar_as_int(neigh_postype.w) * coeff_width + __scalar_as_int(postype.w);
        Scalar lj12 = s_coeffs[typ_pair].x;
        Scalar lj9 = s_coeffs[typ_pair].y;
        Scalar lj6 = s_coeffs[typ_pair].z;
        Scalar lj4 = s_coeffs[typ_pair].w;

        // calculate 1/r^3 and 1/r^6 (FLOPS: 3)
        Scalar r3inv = r2inv * rsqrtf(rsq);
        Scalar r6inv = r3inv * r3inv;
        // calculate the force magnitude / r (FLOPS: 11)
        Scalar forcemag_divr = r6inv * (r2inv * (Scalar(12.0) * lj12  * r6inv + Scalar(9.0) * r3inv * lj9 + Scalar(6.0) * lj6 ) + Scalar(4.0) * lj4);
        // calculate the virial (FLOPS: 3)
        Scalar forcemag_div2r = Scalar(0.5)*forcemag_divr;
        virial[0] += dx.x*dx.x*forcemag_div2r;
        virial[1] += dx.x*dx.y*forcemag_div2r;
        virial[2] += dx.x*dx.z*forcemag_div2r;
        virial[3] += dx.y*dx.y*forcemag_div2r;
        virial[4] += dx.y*dx.z*forcemag_div2r;
        virial[5] += dx.z*dx.z*forcemag_div2r;

        // calculate the pair energy (FLOPS: 8)
        Scalar pair_eng = r6inv * (lj12 * r6inv + lj9 * r3inv + lj6) + lj4 * r2inv * r2inv;

        // add up the force vector components (FLOPS: 7)
        force.x += dx.x * forcemag_divr;
        force.y += dx.y * forcemag_divr;
        force.z += dx.z * forcemag_divr;
        force.w += pair_eng;
        }

    // potential energy per particle must be halved
    force.w *= Scalar(0.5);
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;
    for (int i = 0; i < 6; i++)
        d_virial[i*virial_pitch+idx] = virial[i];
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param d_head_list Indexes for reading \a d_nlist
    \param d_coeffs A \a coeff_width by \a coeff_width matrix of coefficients indexed by type
        pair i,j. The x-component is the lj12 coefficient and the y-, z-, and w-components
                are the lj9, lj6, and lj4 coefficients, respectively.
    \param coeff_width Width of the \a d_coeffs matrix.
    \param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
        force is set to 0
    \param block_size Block size to execute
    \param compute_capability GPU compute capability (20, 30, 35, ...)
    \param max_tex1d_width Maximum width of a linear 1d texture

    \returns Any error code resulting from the kernel launch

    This is just a driver for calcCGCMMForces_kernel, see the documentation for it for more information.
*/
cudaError_t gpu_compute_cgcmm_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const unsigned int *d_head_list,
                                     const Scalar4 *d_coeffs,
                                     const unsigned int size_nlist,
                                     const unsigned int coeff_width,
                                     const Scalar r_cutsq,
                                     const unsigned int block_size)
    {
    assert(d_coeffs);
    assert(coeff_width > 0);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    gpu_compute_cgcmm_forces_kernel<<< grid, threads, sizeof(Scalar4)*coeff_width*coeff_width >>>(d_force,
                                                                                                  d_virial,
                                                                                                  virial_pitch,
                                                                                                  N,
                                                                                                  d_pos,
                                                                                                  box,
                                                                                                  d_n_neigh,
                                                                                                  d_nlist,
                                                                                                  d_head_list,
                                                                                                  d_coeffs,
                                                                                                  coeff_width,
                                                                                                  r_cutsq);

    return cudaSuccess;
    }

// vim:syntax=cpp
