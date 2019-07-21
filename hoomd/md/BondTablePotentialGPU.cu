// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "BondTablePotentialGPU.cuh"
#include "hoomd/TextureTools.h"


#include <assert.h>

/*! \file BondTablePotentialGPU.cu
    \brief Defines GPU kernel code for calculating the table bond forces. Used by BondTablePotentialGPU.
*/

/*!  This kernel is called to calculate the table pair forces on all N particles

    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles in system
    \param d_pos device array of particle positions
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param pitch Pitch of 2D bond list
    \param n_bonds_list List of numbers of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_params Parameters for each table associated with a type pair
    \param table_value index helper function
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be evaluated

    See BondTablePotential for information on the memory layout.
*/
__global__ void gpu_compute_bondtable_forces_kernel(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim box,
                                     const group_storage<2> *blist,
                                     const unsigned int pitch,
                                     const unsigned int *n_bonds_list,
                                     const unsigned int n_bond_type,
                                     const Scalar2 *d_tables,
                                     const Scalar4 *d_params,
                                     const Index2D table_value,
                                     unsigned int *d_flags)
    {


    // read in params for easy and fast access in the kernel
    extern __shared__ Scalar4 s_params[];
    for (unsigned int cur_offset = 0; cur_offset < n_bond_type; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_bond_type)
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
        }
    __syncthreads();


    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_bonds =n_bonds_list[idx];

    // read in the position of our particle.
    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // initialize the force to 0
    Scalar4 force = make_scalar4(0.0f, 0.0f, 0.0f, 0.0f);
    // initialize the virial tensor to 0
    Scalar virial[6];
    for (unsigned int i = 0; i < 6; i++)
        virial[i] = 0;

    // loop over neighbors
    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        // MEM TRANSFER: 8 bytes
        group_storage<2> cur_bond = blist[pitch*bond_idx + idx];

        int cur_bond_idx = cur_bond.idx[0];
        int cur_bond_type = cur_bond.idx[1];

        // get the bonded particle's position (MEM_TRANSFER: 16 bytes)
        Scalar4 neigh_postype = d_pos[cur_bond_idx];
        Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

        // calculate dr (FLOPS: 3)
        Scalar3 dx = pos - neigh_pos;

        // apply periodic boundary conditions (FLOPS: 12)
        dx = box.minImage(dx);

        // access needed parameters
        Scalar4 params = s_params[cur_bond_type];
        Scalar rmin = params.x;
        Scalar rmax = params.y;
        Scalar delta_r = params.z;

        // calculate r
        Scalar rsq = dot(dx, dx);
        Scalar r = sqrtf(rsq);

        if (r < rmax && r >= rmin)
            {
            // precomputed term
            Scalar value_f = (r - rmin) / delta_r;

            // compute index into the table and read in values
            unsigned int value_i = floor(value_f);

            Scalar2 VF0 = __ldg(d_tables + table_value(value_i, cur_bond_type));
            Scalar2 VF1 = __ldg(d_tables + table_value(value_i+1, cur_bond_type));
            // unpack the data
            Scalar V0 = VF0.x;
            Scalar V1 = VF1.x;
            Scalar F0 = VF0.y;
            Scalar F1 = VF1.y;

            // compute the linear interpolation coefficient
            Scalar f = value_f - Scalar(value_i);

            // interpolate to get V and F;
            Scalar V = V0 + f * (V1 - V0);
            Scalar F = F0 + f * (F1 - F0);

            // convert to standard variables used by the other pair computes in HOOMD-blue
            Scalar forcemag_divr = 0.0f;
            if (r > 0.0f)
                forcemag_divr = F / r;
            Scalar bond_eng = V;
            // calculate the virial
            Scalar force_div2r = Scalar(0.5) * forcemag_divr;
            virial[0] += dx.x * dx.x * force_div2r; // xx
            virial[1] += dx.x * dx.y * force_div2r; // xy
            virial[2] += dx.x * dx.z * force_div2r; // xz
            virial[3] += dx.y * dx.y * force_div2r; // yy
            virial[4] += dx.y * dx.z * force_div2r; // yz
            virial[5] += dx.z * dx.z * force_div2r; // zz

            // add up the force vector components (FLOPS: 7)
            force.x += dx.x * forcemag_divr;
            force.y += dx.y * forcemag_divr;
            force.z += dx.z * forcemag_divr;
            force.w += bond_eng * 0.5f;
            }
        else
            {
            *d_flags = 1;
            }
        }


    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;
    for (unsigned int i = 0; i < 6 ; i++)
        d_virial[i*virial_pitch + idx] = virial[i];
    }


/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions used to implement periodic boundary conditions
    \param blist List of bonds stored on the GPU
    \param pitch Pitch of 2D bond list
    \param n_bonds_list List of numbers of bonds stored on the GPU
    \param n_bond_type number of bond types
    \param d_tables Tables of the potential and force
    \param d_params Parameters for each table associated with a type pair
    \param table_width Number of entries in the table
    \param table_value indexer helper
    \param d_flags flags on the device - a 1 will be written if evaluation
                   of forces failed for any bond
    \param block_size Block size at which to run the kernel

    \note This is just a kernel driver. See gpu_compute_bondtable_forces_kernel for full documentation.
*/
cudaError_t gpu_compute_bondtable_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim &box,
                                     const group_storage<2> *blist,
                                     const unsigned int pitch,
                                     const unsigned int *n_bonds_list,
                                     const unsigned int n_bond_type,
                                     const Scalar2 *d_tables,
                                     const Scalar4 *d_params,
                                     const unsigned int table_width,
                                     const Index2D &table_value,
                                     unsigned int *d_flags,
                                     const unsigned int block_size)
    {
    assert(d_params);
    assert(d_tables);
    assert(n_bond_type > 0);
    assert(table_width > 1);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_bondtable_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    gpu_compute_bondtable_forces_kernel<<< grid, threads, sizeof(Scalar4)*n_bond_type >>>
            (d_force,
             d_virial,
             virial_pitch,
             N,
             d_pos,
             box,
             blist,
             pitch,
             n_bonds_list,
             n_bond_type,
             d_tables,
             d_params,
             table_value,
             d_flags);

    return cudaSuccess;
    }

// vim:syntax=cpp
