// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ConstantForceComputeGPU.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file ConstantForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating constant forces forces on the GPU. Used by
   ConstantForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for setting constant force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_force particle force on device
    \param d_torque particle torque on device
    \param d_f_const particle constant force unit vector
    \param d_t_const particle constant torque unit vector
    \param orientationLink check if particle orientation is linked to constant force vector
*/
__global__ void gpu_compute_constant_force_set_forces_kernel(const unsigned int group_size,
                                                             unsigned int* d_index_array,
                                                             Scalar4* d_force,
                                                             Scalar4* d_torque,
                                                             const Scalar4* d_pos,
                                                             const Scalar3* d_f_const,
                                                             const Scalar3* d_t_const,
                                                             const unsigned int N)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    Scalar3 fconst = d_f_const[type];

    vec3<Scalar> fi(fconst.x, fconst.y, fconst.z);
    d_force[idx] = vec_to_scalar4(fi, 0);

    Scalar3 tconst = d_t_const[type];
    vec3<Scalar> ti(tconst.x, tconst.y, tconst.z);
    d_torque[idx] = vec_to_scalar4(ti, 0);
    }

hipError_t gpu_compute_constant_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 Scalar4* d_torque,
                                                 const Scalar4* d_pos,
                                                 const Scalar3* d_f_const,
                                                 const Scalar3* d_t_const,
                                                 const unsigned int N,
                                                 unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipMemset(d_force, 0, sizeof(Scalar4) * N);
    hipMemset(d_torque, 0, sizeof(Scalar4) * N);
    hipLaunchKernelGGL((gpu_compute_constant_force_set_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_index_array,
                       d_force,
                       d_torque,
                       d_pos,
                       d_f_const,
                       d_t_const,
                       N);
    return hipSuccess;
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
