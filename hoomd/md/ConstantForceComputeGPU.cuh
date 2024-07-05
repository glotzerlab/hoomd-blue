// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ConstantForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating constant forces forces on the GPU. Used by
   ConstantForceComputeGPU.
*/

#ifndef __CONSTANT_FORCE_COMPUTE_GPU_CUH__
#define __CONSTANT_FORCE_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_constant_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 Scalar4* d_torque,
                                                 const Scalar4* d_pos,
                                                 const Scalar3* d_f_act,
                                                 const Scalar3* d_t_act,
                                                 const unsigned int N,
                                                 unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
