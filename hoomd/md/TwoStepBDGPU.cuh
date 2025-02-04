// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepBDGPU.cuh
    \brief Declares GPU kernel code for Brownian dynamics on the GPU. Used by TwoStepBDGPU.
*/

#include "TwoStepLangevinGPU.cuh"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __TWO_STEP_BD_GPU_CUH__
#define __TWO_STEP_BD_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the first part of the Brownian update called by TwoStepBDGPU
hipError_t gpu_brownian_step_one(Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 int3* d_image,
                                 const BoxDim& box,
                                 const unsigned int* d_tag,
                                 const unsigned int* d_group_members,
                                 const unsigned int group_size,
                                 const Scalar4* d_net_force,
                                 const Scalar3* d_gamma_r,
                                 Scalar4* d_orientation,
                                 Scalar4* d_torque,
                                 const Scalar3* d_inertia,
                                 Scalar4* d_angmom,
                                 const langevin_step_two_args& langevin_args,
                                 const bool aniso,
                                 const Scalar deltaT,
                                 const unsigned int D,
                                 const bool d_noiseless_t,
                                 const bool d_noiseless_r,
                                 const GPUPartition& gpu_partition);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_BD_GPU_CUH__
