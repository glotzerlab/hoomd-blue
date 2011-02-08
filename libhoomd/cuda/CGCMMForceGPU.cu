/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: akohlmey

#include "CGCMMForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file CGCMMForceGPU.cu
    \brief Defines GPU kernel code for calculating the Lennard-Jones pair forces. Used by CGCMMForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Kernel for calculating CG-CMM Lennard-Jones forces
/*! This kernel is called to calculate the Lennard-Jones forces on all N particles for the CG-CMM model potential.

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data on the GPU to calculate forces on
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param nli Indexer for indexing \a d_nlist
    \param d_coeffs Coefficients to the lennard jones force (lj12, lj9, lj6, lj4).
    \param coeff_width Width of the coefficient matrix
    \param r_cutsq Precalculated r_cut*r_cut, where r_cut is the radius beyond which forces are
    set to 0
    \param box Box dimensions used to implement periodic boundary conditions

    \a coeffs is a pointer to a matrix in memory. \c coeffs[i*coeff_width+j].x is \a lj12 for the type pair \a i, \a j.
    Similarly, .y, .z, and .w are the \a lj9, \a lj6, and \a lj4 parameters, respectively. The values in d_coeffs are
    read into shared memory, so \c coeff_width*coeff_width*sizeof(float4) bytes of extern shared memory must be allocated
    for the kernel call.

    Developer information:
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
__global__ void gpu_compute_cgcmm_forces_kernel(float4* d_force,
                                                float* d_virial,
                                               const gpu_pdata_arrays pdata,
                                               const gpu_boxsize box,
                                               const unsigned int *d_n_neigh,
                                               const unsigned int *d_nlist,
                                               const Index2D nli,
                                               const float4 *d_coeffs,
                                               const int coeff_width,
                                               const float r_cutsq)
    {
    // read in the coefficients
    extern __shared__ float4 s_coeffs[];
    for (unsigned int cur_offset = 0; cur_offset < coeff_width*coeff_width; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < coeff_width*coeff_width)
            s_coeffs[cur_offset + threadIdx.x] = d_coeffs[cur_offset + threadIdx.x];
        }
    __syncthreads();
    
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= pdata.N)
        return;
    
    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = d_n_neigh[idx];
    
    // read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    float4 pos = tex1Dfetch(pdata_pos_tex, idx);
    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virial = 0.0f;
    
    // prefetch neighbor index
    unsigned int cur_neigh = 0;
    unsigned int next_neigh = d_nlist[nli(idx, 0)];
    
    // loop over neighbors
    // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    #if (__CUDA_ARCH__ < 200)
    for (int neigh_idx = 0; neigh_idx < nli.getH(); neigh_idx++)
    #else
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
    #endif
        {
        #if (__CUDA_ARCH__ < 200)
        if (neigh_idx < n_neigh)
        #endif
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_neigh = next_neigh;
            next_neigh = d_nlist[nli(idx, neigh_idx+1)];
                
            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);
            
            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float dx = pos.x - neigh_pos.x;
            float dy = pos.y - neigh_pos.y;
            float dz = pos.z - neigh_pos.z;
            
            // apply periodic boundary conditions: (FLOPS 12)
            dx -= box.Lx * rintf(dx * box.Lxinv);
            dy -= box.Ly * rintf(dy * box.Lyinv);
            dz -= box.Lz * rintf(dz * box.Lzinv);
            
            // calculate r squard (FLOPS: 5)
            float rsq = dx*dx + dy*dy + dz*dz;
            
            // calculate 1/r^2 (FLOPS: 2)
            float r2inv;
            if (rsq >= r_cutsq)
                r2inv = 0.0f;
            else
                r2inv = 1.0f / rsq;
                
            // lookup the coefficients between this combination of particle types
            int typ_pair = __float_as_int(neigh_pos.w) * coeff_width + __float_as_int(pos.w);
            float lj12 = s_coeffs[typ_pair].x;
            float lj9 = s_coeffs[typ_pair].y;
            float lj6 = s_coeffs[typ_pair].z;
            float lj4 = s_coeffs[typ_pair].w;
            
            // calculate 1/r^3 and 1/r^6 (FLOPS: 3)
            float r3inv = r2inv * rsqrtf(rsq);
            float r6inv = r3inv * r3inv;
            // calculate the force magnitude / r (FLOPS: 11)
            float forcemag_divr = r6inv * (r2inv * (12.0f * lj12  * r6inv + 9.0f * r3inv * lj9 + 6.0f * lj6 ) + 4.0f * lj4);
            // calculate the virial (FLOPS: 3)
            virial += float(1.0/6.0) * rsq * forcemag_divr;
            // calculate the pair energy (FLOPS: 8)
            float pair_eng = r6inv * (lj12 * r6inv + lj9 * r3inv + lj6) + lj4 * r2inv * r2inv;
            
            // add up the force vector components (FLOPS: 7)
            force.x += dx * forcemag_divr;
            force.y += dy * forcemag_divr;
            force.z += dz * forcemag_divr;
            force.w += pair_eng;
            }
        }
        
    // potential energy per particle must be halved
    force.w *= 0.5f;
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;
    d_virial[idx] = virial;
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param d_n_neigh Device memory array listing the number of neighbors for each particle
    \param d_nlist Device memory array containing the neighbor list contents
    \param nli Indexer for indexing \a d_nlist
    \param d_coeffs A \a coeff_width by \a coeff_width matrix of coefficients indexed by type
        pair i,j. The x-component is the lj12 coefficient and the y-, z-, and w-components
                are the lj9, lj6, and lj4 coefficients, respectively.
    \param coeff_width Width of the \a d_coeffs matrix.
    \param r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
        force is set to 0
    \param block_size Block size to execute

    \returns Any error code resulting from the kernel launch

    This is just a driver for calcCGCMMForces_kernel, see the documentation for it for more information.
*/
cudaError_t gpu_compute_cgcmm_forces(float4* d_force,
                                     float* d_virial,
                                     const gpu_pdata_arrays &pdata,
                                     const gpu_boxsize &box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const Index2D& nli,
                                     const float4 *d_coeffs,
                                     const unsigned int coeff_width,
                                     const float r_cutsq,
                                     const unsigned int block_size)
    {
    assert(d_coeffs);
    assert(coeff_width > 0);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the texture
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_compute_cgcmm_forces_kernel<<< grid, threads, sizeof(float4)*coeff_width*coeff_width >>>
         (d_force, d_virial, pdata, box, d_n_neigh, d_nlist, nli, d_coeffs, coeff_width, r_cutsq);
        
    return cudaSuccess;
    }

// vim:syntax=cpp

