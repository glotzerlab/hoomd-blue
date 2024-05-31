// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "SurfaceTensionMeshForceComputeGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file SurfaceTensionMeshForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating the surface tension forces. Used by
   SurfaceTensionMeshForceComputeComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void
gpu_compute_surface_tension_kernel(Scalar* d_partial_sum_area,
                                                 const unsigned int N,
                                                 const Scalar4* d_pos,
                                                 BoxDim box,
                                                 const group_storage<3>* tlist,
                                                 const Index2D tlist_idx,
                                                 const unsigned int* n_triangles_list)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* area_sdata = (Scalar*)&s_data[0];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar area_transfer = 0;

    if (idx < N)
        {
        int n_triangles = n_triangles_list[idx];
        Scalar4 postype = __ldg(d_pos + idx);
        Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

        for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
            {
            group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

            int cur_mem2_idx = cur_triangle.idx[0];
            int cur_mem3_idx = cur_triangle.idx[1];

            // get the b-particle's position (MEM TRANSFER: 16 bytes)
            Scalar4 bb_postype = d_pos[cur_mem2_idx];
            Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);

            // get the c-particle's position (MEM TRANSFER: 16 bytes)
            Scalar4 cc_postype = d_pos[cur_mem3_idx];
            Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);

            Scalar3 dab = pos_b - pos_a;
            Scalar3 dac = pos_c - pos_a;

            dab = box.minImage(dab);
            dac = box.minImage(dac);

            Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
            Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;

            Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dab.z * dac.z;

            Scalar Area_3 = sqrt(rsqab * rsqac - rabrac * rabrac) / 6.0;

            area_transfer += Area_3;
            }
        }

    area_sdata[threadIdx.x] = area_transfer;
    __syncthreads();

    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            area_sdata[threadIdx.x] += area_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }

    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum_area[blockIdx.x] = area_sdata[0];
        }
    }

//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the partial sum
    \param num_blocks Number of blocks to execute
*/
__global__ void
gpu_triangle_area_reduce_partial_sum_kernel(Scalar* d_sum, Scalar* d_partial_sum, unsigned int num_blocks)
    {
    Scalar sum = Scalar(0.0);
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* area_sdata = (Scalar*)&s_data[0];

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            area_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            area_sdata[threadIdx.x] = Scalar(0.0);
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                area_sdata[threadIdx.x] += area_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        // everybody sums up sum2K
        sum += area_sdata[0];
        }

    if (threadIdx.x == 0)
        *d_sum = sum;
    }

/*! \param d_sigma Device memory to write per paricle sigma
    \param d_sigma_dash Device memory to write per particle sigma_dash
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_rtag device array of particle reverse tags
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param blist List of mesh bonds stored on the GPU
    \param d_triangles device array of mesh triangles
    \param n_bonds_list List of numbers of mesh bonds stored on the GPU
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_surface_tension(Scalar* d_sum_area,
                                                     Scalar* d_sum_partial_area,
                                                     const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const BoxDim& box,
                                                     const group_storage<3>* tlist,
                                                     const Index2D tlist_idx,
                                                     const unsigned int* n_triangles_list,
                                                     unsigned int block_size,
                                                     unsigned int num_blocks)
    {
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_surface_tension_kernel),
                       dim3(grid),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_sum_partial_area,
                       N,
                       d_pos,
                       box,
                       tlist,
                       tlist_idx,
                       n_triangles_list);

    hipLaunchKernelGGL((gpu_triangle_area_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       &d_sum_area[0],
                       d_sum_partial_area,
                       num_blocks);

    return hipSuccess;
    }

//! Kernel for calculating area conservation force on the GPU
/*!
    \param d_area Device memory to write total surface area
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params K,A_mesh params packed as Scalar variables
    \param n_triangle_type number of mesh triangle types
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
*/
__global__ void
gpu_compute_surface_tension_force_kernel(Scalar4* d_force,
                                                  Scalar* d_virial,
                                                  const size_t virial_pitch,
                                                  const unsigned int N,
                                                  const unsigned int N_tri,
                                                  const Scalar4* d_pos,
                                                  BoxDim box,
                                                  const group_storage<3>* tlist,
                                                  const unsigned int* tpos_list,
                                                  const Index2D tlist_idx,
                                                  const unsigned int* n_triangles_list,
                                                  Scalar* d_params,
                                                  const unsigned int n_triangle_type,
                                                  unsigned int* d_flags)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_triangles = n_triangles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 postype = __ldg(d_pos + idx);
    Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    // Scalar area = 0.0;
    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_abc = tpos_list[tlist_idx(idx, triangle_idx)];

        int cur_mem2_idx = cur_triangle.idx[0];
        int cur_mem3_idx = cur_triangle.idx[1];
        int cur_triangle_type = cur_triangle.idx[2];

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_mem2_idx];
        Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);

        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_mem3_idx];
        Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);

        Scalar3 dab, dac;
        if (cur_triangle_abc == 0)
            {
            dab = pos_b - pos_a;
            dac = pos_c - pos_a;
            }
        else
            {
            if (cur_triangle_abc == 1)
                {
                dab = pos_a - pos_b;
                dac = pos_c - pos_b;
                }
            else
                {
                dab = pos_c - pos_b;
                dac = pos_a - pos_b;
                }
            }

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        Scalar sigma = __ldg(d_params + cur_triangle_type);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars

        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;

        Scalar rabrac = dab.x * dac.x + dab.y * dac.y + dab.z * dac.z;

        Scalar area2 = sqrt(rsqab * rsqac - rabrac * rabrac);

        Scalar prefactor = -sigma / (2 * area2);

        Scalar energy_pp = sigma * area2 / 2;

        Scalar3 Fa;

        if (cur_triangle_abc == 0)
            {
            Fa = (rabrac - rsqac) * dab + (rabrac - rsqab) * dac;
            }
        else
            {
            if (cur_triangle_abc == 1)
                {
                Fa = rsqac * dab - rabrac * dac;
                }
            else
                {
                Fa = rsqab * dac - rabrac * dab;
                }
            }

        Fa = prefactor * Fa;

        force.x += Fa.x;
        force.y += Fa.y;
        force.z += Fa.z;
        force.w += energy_pp; // divided by 3 because of three
                                                                 // particles sharing the energy

        virial[0] += Scalar(1. / 2.) * pos_a.x * Fa.x; // xx
        virial[1] += Scalar(1. / 2.) * pos_a.y * Fa.x; // xy
        virial[2] += Scalar(1. / 2.) * pos_a.z * Fa.x; // xz
        virial[3] += Scalar(1. / 2.) * pos_a.y * Fa.y; // yy
        virial[4] += Scalar(1. / 2.) * pos_a.z * Fa.y; // yz
        virial[5] += Scalar(1. / 2.) * pos_a.z * Fa.z; // zz
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force;

    for (unsigned int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = virial[i];
    }

/*!
    \param d_area Device memory to write total surface area
    \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangles stored on the GPU
    \param n_triangles_list List of numbers of mesh triangles stored on the GPU
    \param d_params sigma params packed as Scalar variable
    \param n_triangle_type number of mesh triangle types
    \param block_size Block size to use when performing calculations
    \param d_flags Flag allocated on the device for use in checking for bonds that cannot be
    \param compute_capability Device compute capability (200, 300, 350, ...)
    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_surface_tension_force(Scalar4* d_force,
                                                      Scalar* d_virial,
                                                      const size_t virial_pitch,
                                                      const unsigned int N,
                                                      const unsigned int N_tri,
                                                      const Scalar4* d_pos,
                                                      const BoxDim& box,
                                                      const group_storage<3>* tlist,
                                                      const unsigned int* tpos_list,
                                                      const Index2D tlist_idx,
                                                      const unsigned int* n_triangles_list,
                                                      Scalar* d_params,
                                                      const unsigned int n_triangle_type,
                                                      int block_size,
                                                      unsigned int* d_flags)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_surface_tension_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_surface_tension_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       N_tri,
                       d_pos,
                       box,
                       tlist,
                       tpos_list,
                       tlist_idx,
                       n_triangles_list,
                       d_params,
                       n_triangle_type,
                       d_flags);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
