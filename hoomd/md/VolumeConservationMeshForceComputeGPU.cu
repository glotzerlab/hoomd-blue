// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "VolumeConservationMeshForceComputeGPU.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#include <assert.h>

#include <stdio.h>

/*! \file MeshVolumeConservationGPU.cu
    \brief Defines GPU kernel code for calculating the volume_constraint forces. Used by
   MeshVolumeConservationComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for calculating volume_constraint sigmas on the GPU
/*! \param d_partial_sum_volume Device memory to write partial meah volume
    \param N number of particles
    \param tN number of mesh types
    \param cN current mesh type index
    \param d_pos device array of particle positions
    \param d_image device array of particle images
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangle indices stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param ignore_type ignores mesh type if true
    \param n_triangles_list List of mesh triangles stored on the GPU
*/
__global__ void gpu_compute_volume_constraint_volume_kernel(Scalar* d_partial_sum_volume,
                                                            const unsigned int N,
                                                            const unsigned int tN,
                                                            const unsigned int cN,
                                                            const Scalar4* d_pos,
                                                            const int3* d_image,
                                                            BoxDim box,
                                                            const group_storage<3>* tlist,
                                                            const unsigned int* tpos_list,
                                                            const Index2D tlist_idx,
                                             		    const bool ignore_type,
                                                            const unsigned int* n_triangles_list)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* volume_sdata = (Scalar*)&s_data[0];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Scalar volume_transfer = 0;

    if (idx < N)
        {
        int n_triangles = n_triangles_list[idx];
        Scalar4 postype = __ldg(d_pos + idx);
        Scalar3 pos_a = make_scalar3(postype.x, postype.y, postype.z);
        int3 image_a = d_image[idx];
        pos_a = box.shift(pos_a, image_a);

        for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
            {
            group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];
            int cur_triangle_type = cur_triangle.idx[2];

            if(ignore_type) cur_triangle_type = 0;

	    if(cur_triangle_type != cN) continue;

            int cur_triangle_b = cur_triangle.idx[0];
            int cur_triangle_c = cur_triangle.idx[1];

            int cur_triangle_abc = tpos_list[tlist_idx(idx, triangle_idx)];

            // get the b-particle's position (MEM TRANSFER: 16 bytes)
            Scalar4 bb_postype = d_pos[cur_triangle_b];
            Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
            int3 image_b = d_image[cur_triangle_b];
            pos_b = box.shift(pos_b, image_b);

            // get the c-particle's position (MEM TRANSFER: 16 bytes)
            Scalar4 cc_postype = d_pos[cur_triangle_c];
            Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);
            int3 image_c = d_image[cur_triangle_c];
            pos_c = box.shift(pos_c, image_c);

            vec3<Scalar> dVol(0, 0, 0);
            if (cur_triangle_abc == 1)
                {
                dVol.x = pos_b.y * pos_c.z - pos_b.z * pos_c.y;
                dVol.y = pos_b.z * pos_c.x - pos_b.x * pos_c.z;
                dVol.z = pos_b.x * pos_c.y - pos_b.y * pos_c.x;
                }
            else
                {
                dVol.x = pos_c.y * pos_b.z - pos_c.z * pos_b.y;
                dVol.y = pos_c.z * pos_b.x - pos_c.x * pos_b.z;
                dVol.z = pos_c.x * pos_b.y - pos_c.y * pos_b.x;
                }
            Scalar Vol = dVol.x * pos_a.x + dVol.y * pos_a.y + dVol.z * pos_a.z;
            volume_transfer += Vol / 18.0;
            }
        }

    volume_sdata[threadIdx.x] = volume_transfer;//[i_types];
     
    __syncthreads();
     
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
    	{
    	volume_sdata[threadIdx.x] += volume_sdata[threadIdx.x + offs];
    	}
        offs >>= 1;
        __syncthreads();
        }
    
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        //d_partial_sum_volume[blockIdx.x * tN + i_types] = volume_sdata[0];
        d_partial_sum_volume[blockIdx.x * tN + cN] = volume_sdata[0];
        }
    }

//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the partial sum
    \param num_blocks Number of blocks to execute
*/
__global__ void gpu_volume_reduce_partial_sum_kernel(Scalar* d_sum,
                                                     Scalar* d_partial_sum,
                                                     unsigned int tN,
                                                     unsigned int num_blocks)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* volume_sdata = (Scalar*)&s_data[0];

    for (unsigned int i_types = 0; i_types < tN; i_types++)
        {
        // sum up the values in the partial sum via a sliding window
        Scalar sum = Scalar(0.0);
        for (int start = 0; start < num_blocks; start += blockDim.x)
            {
            __syncthreads();
            if (start + threadIdx.x < num_blocks)
                volume_sdata[threadIdx.x] = d_partial_sum[(start + threadIdx.x) * tN + i_types];
            else
                volume_sdata[threadIdx.x] = Scalar(0.0);
            __syncthreads();

            // reduce the sum in parallel
            int offs = blockDim.x >> 1;
            while (offs > 0)
                {
                if (threadIdx.x < offs)
                    volume_sdata[threadIdx.x] += volume_sdata[threadIdx.x + offs];
                offs >>= 1;
                }

            // everybody sums up sum2K
            sum += volume_sdata[0];
            }

        if (threadIdx.x == 0)
            d_sum[i_types] = sum;
        }
    }

/*! \param d_partial_sum_volume Device memory to write partial meah volume
    \param N number of particles
    \param tN number of mesh types
    \param cN current mesh type index
    \param d_pos device array of particle positions
    \param d_image device array of particle images
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist List of mesh triangle indices stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param ignore_type ignores mesh type if true
    \param n_triangles_list List of mesh triangles stored on the GPU
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_volume_constraint_volume(Scalar* d_sum_volume,
                                                Scalar* d_sum_partial_volume,
                                                const unsigned int N,
                                                const unsigned int tN,
                                                const Scalar4* d_pos,
                                                const int3* d_image,
                                                const BoxDim& box,
                                                const group_storage<3>* tlist,
                                                const unsigned int* tpos_list,
                                                const Index2D tlist_idx,
                                                const bool ignore_type,
                                                const unsigned int* n_triangles_list,
                                                unsigned int block_size,
                                                unsigned int num_blocks)
    {
    dim3 grid(num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(block_size, 1, 1);

    for (unsigned int i_types = 0; i_types < tN; i_types++)
        {
        // run the kernel
        hipLaunchKernelGGL((gpu_compute_volume_constraint_volume_kernel),
                           dim3(grid),
                           dim3(threads),
                           block_size * sizeof(Scalar),
                           0,
                           d_sum_partial_volume,
                           N,
                           tN,
			   i_types,
                           d_pos,
                           d_image,
                           box,
                           tlist,
                           tpos_list,
                           tlist_idx,
                           ignore_type,
                           n_triangles_list);
        }

    hipLaunchKernelGGL((gpu_volume_reduce_partial_sum_kernel),
                       dim3(grid1),
                       dim3(threads),
                       block_size * sizeof(Scalar),
                       0,
                       d_sum_volume,
                       d_sum_partial_volume,
                       tN,
                       num_blocks);

    return hipSuccess;
    }

//! Kernel for calculating volume_constraint sigmas on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch
    \param N Number of particles
    \param gN Number of triangles of a triangle type
    \param aN Total global number of triangles
    \param d_pos device array of particle positions
    \param d_magep device array of particle images
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param area Total instantaneous area per mesh type
    \param tlist List of mesh triangle indices stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param n_triangles_list total group number of triangles
    \param d_params K, V0 params packed as Scalar variables
    \param ignore_type ignores mesh type if true
*/
__global__ void gpu_compute_volume_constraint_force_kernel(Scalar4* d_force,
                                                           Scalar* d_virial,
                                                           const size_t virial_pitch,
                                                           const unsigned int N,
                                                           const unsigned int* gN,
                                               		   const unsigned int aN,
                                                           const Scalar4* d_pos,
                                                           const int3* d_image,
                                                           BoxDim box,
                                                           const Scalar* volume,
                                                           const group_storage<3>* tlist,
                                                           const unsigned int* tpos_list,
                                                           const Index2D tlist_idx,
                                                           const unsigned int* n_triangles_list,
                                                           Scalar2* d_params,
                                                           const bool ignore_type)
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
    int3 image_a = d_image[idx];
    pos_a = box.shift(pos_a, image_a);

    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // initialize the virial to 0
    Scalar virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = Scalar(0.0);

    unsigned int triN = 1*aN;

    // loop over all triangles
    for (int triangle_idx = 0; triangle_idx < n_triangles; triangle_idx++)
        {
        group_storage<3> cur_triangle = tlist[tlist_idx(idx, triangle_idx)];

        int cur_triangle_b = cur_triangle.idx[0];
        int cur_triangle_c = cur_triangle.idx[1];
        int cur_triangle_type = cur_triangle.idx[2];

	if(ignore_type) cur_triangle_type = 0;
	else triN = gN[cur_triangle_type];

        // get the angle parameters (MEM TRANSFER: 8 bytes)
        Scalar2 params = __ldg(d_params + cur_triangle_type);
        Scalar K = params.x;
        Scalar V0 = params.y;

        Scalar VolDiff = volume[cur_triangle_type] - V0;

        Scalar energy = K * VolDiff * VolDiff / (6 * V0 * triN);

        VolDiff = -K / V0 * VolDiff / 6.0;

        int cur_triangle_abc = tpos_list[tlist_idx(idx, triangle_idx)];

        // get the b-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 bb_postype = d_pos[cur_triangle_b];
        Scalar3 pos_b = make_scalar3(bb_postype.x, bb_postype.y, bb_postype.z);
        int3 image_b = d_image[cur_triangle_b];
        pos_b = box.shift(pos_b, image_b);

        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 cc_postype = d_pos[cur_triangle_c];
        Scalar3 pos_c = make_scalar3(cc_postype.x, cc_postype.y, cc_postype.z);
        int3 image_c = d_image[cur_triangle_c];
        pos_c = box.shift(pos_c, image_c);

        vec3<Scalar> dVol;
        if (cur_triangle_abc == 1)
            {
            dVol.x = pos_b.y * pos_c.z - pos_b.z * pos_c.y;
            dVol.y = pos_b.z * pos_c.x - pos_b.x * pos_c.z;
            dVol.z = pos_b.x * pos_c.y - pos_b.y * pos_c.x;
            }
        else
            {
            dVol.x = pos_c.y * pos_b.z - pos_c.z * pos_b.y;
            dVol.y = pos_c.z * pos_b.x - pos_c.x * pos_b.z;
            dVol.z = pos_c.x * pos_b.y - pos_c.y * pos_b.x;
            }

        Scalar3 Fa;

        Fa.x = VolDiff * dVol.x;
        Fa.y = VolDiff * dVol.y;
        Fa.z = VolDiff * dVol.z;

        force.x += Fa.x;
        force.y += Fa.y;
        force.z += Fa.z;
        force.w += energy;

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

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param N Number of particles
    \param gN Number of triangles of a triangle type
    \param aN Total global number of triangles
    \param d_pos device array of particle positions
    \param d_magep device array of particle images
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param area Total instantaneous area per mesh type
    \param tlist List of mesh triangle indices stored on the GPU
    \param tpos_list Position of current index in list of mesh triangles stored on the GPU
    \param n_triangles_list total group number of triangles
    \param d_params K, V0 params packed as Scalar variables
    \param ignore_type ignores mesh type if true
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()
*/
hipError_t gpu_compute_volume_constraint_force(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const unsigned int* gN,
                                               const unsigned int aN,
                                               const Scalar4* d_pos,
                                               const int3* d_image,
                                               const BoxDim& box,
                                               const Scalar* volume,
                                               const group_storage<3>* tlist,
                                               const unsigned int* tpos_list,
                                               const Index2D tlist_idx,
                                               const unsigned int* n_triangles_list,
                                               Scalar2* d_params,
                                               const bool ignore_type,
                                               int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_volume_constraint_force_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_volume_constraint_force_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       gN,
		       aN,
                       d_pos,
                       d_image,
                       box,
                       volume,
                       tlist,
                       tpos_list,
                       tlist_idx,
                       n_triangles_list,
                       d_params,
                       ignore_type);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
