// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepNVEGPU.cuh
    \brief Declares GPU kernel code for NVE integration on the GPU. Used by TwoStepNVEGPU.
*/

#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __TWO_STEP_NVE_GPU_CUH__
#define __TWO_STEP_NVE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the first part of the NVE update called by TwoStepNVEGPU
hipError_t gpu_nve_step_one(Scalar4* d_pos,
                            Scalar4* d_vel,
                            const Scalar3* d_accel,
                            int3* d_image,
                            unsigned int* d_group_members,
                            const GPUPartition& gpu_partition,
                            const BoxDim& box,
                            Scalar deltaT,
                            bool limit,
                            Scalar limit_val,
                            bool zero_force,
                            unsigned int block_size);

//! Kernel driver for the second part of the NVE update called by TwoStepNVEGPU
hipError_t gpu_nve_step_two(Scalar4* d_vel,
                            Scalar3* d_accel,
                            unsigned int* d_group_members,
                            const GPUPartition& gpu_partition,
                            Scalar4* d_net_force,
                            Scalar deltaT,
                            bool limit,
                            Scalar limit_val,
                            bool zero_force,
                            unsigned int block_size);

//! Kernel driver for the first part of the angular NVE update (NO_SQUISH) by TwoStepNVEPU
hipError_t gpu_nve_angular_step_one(Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const Scalar4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const GPUPartition& gpu_partition,
                                    Scalar deltaT,
                                    Scalar scale,
                                    const unsigned int block_size);

//! Kernel driver for the second part of the angular NVE update (NO_SQUISH) by TwoStepNVEPU
hipError_t gpu_nve_angular_step_two(const Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const Scalar4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const GPUPartition& gpu_partition,
                                    Scalar deltaT,
                                    Scalar scale,
                                    const unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_NVE_GPU_CUH__
