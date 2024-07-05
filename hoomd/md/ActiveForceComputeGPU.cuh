// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ActiveForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by
   ActiveForceComputeGPU.
*/

#ifndef __ACTIVE_FORCE_COMPUTE_GPU_CUH__
#define __ACTIVE_FORCE_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_active_force_set_forces(const unsigned int group_size,
                                               unsigned int* d_index_array,
                                               Scalar4* d_force,
                                               Scalar4* d_torque,
                                               const Scalar4* d_pos,
                                               const Scalar4* d_orientation,
                                               const Scalar4* d_f_act,
                                               const Scalar4* d_t_act,
                                               const unsigned int N,
                                               unsigned int block_size);

hipError_t gpu_compute_active_force_rotational_diffusion(const unsigned int group_size,
                                                         unsigned int* d_tag,
                                                         unsigned int* d_index_array,
                                                         const Scalar4* d_pos,
                                                         Scalar4* d_orientation,
                                                         const Scalar4* d_f_act,
                                                         bool is2D,
                                                         const Scalar rotationDiff,
                                                         const uint64_t timestep,
                                                         const uint16_t seed,
                                                         unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
