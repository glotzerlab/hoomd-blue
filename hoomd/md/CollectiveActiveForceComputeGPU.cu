// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CollectiveActiveForceComputeGPU.cuh"
#include "hoomd/TextureTools.h"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file CollectiveActiveForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating collective active forces on the GPU. Used by
   CollectiveActiveForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for setting the collective active force of each particle type on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to particle index
    \param d_force particle force on device
    \param d_pos particle positions on device, whose w component holds the particle type
    \param d_f_act active force unit vector (x, y, z) and magnitude (w) for each particle type
*/
__global__ void gpu_compute_collective_active_force_set_forces_kernel(
    const unsigned int group_size,
    unsigned int* d_index_array,
    Scalar4* d_force,
    const Scalar4* d_pos,
    const Scalar4* d_f_act)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    Scalar4 fact = __ldg(d_f_act + type);
    vec3<Scalar> f(fact.w * fact.x, fact.w * fact.y, fact.w * fact.z);
    d_force[idx] = vec_to_scalar4(f, 0);
    }

hipError_t gpu_compute_collective_active_force_set_forces(const unsigned int group_size,
                                                          unsigned int* d_index_array,
                                                          Scalar4* d_force,
                                                          const Scalar4* d_pos,
                                                          const Scalar4* d_f_act,
                                                          unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_collective_active_force_set_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_index_array,
                       d_force,
                       d_pos,
                       d_f_act);
    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
