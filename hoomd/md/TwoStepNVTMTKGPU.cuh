// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepNVTGPU.cuh
    \brief Declares GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
*/

#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __TWO_STEP_NVT_MTK_GPU_CUH__
#define __TWO_STEP_NVT_MTK_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the first part of the NVT update called by TwoStepNVTGPU
hipError_t gpu_nvt_mtk_step_one(Scalar4* d_pos,
                                Scalar4* d_vel,
                                const Scalar3* d_accel,
                                int3* d_image,
                                unsigned int* d_group_members,
                                unsigned int group_size,
                                const BoxDim& box,
                                unsigned int block_size,
                                Scalar exp_fac,
                                Scalar deltaT,
                                const GPUPartition& gpu_partition);

//! Kernel driver for the second part of the NVT update called by NVTUpdaterGPU
hipError_t gpu_nvt_mtk_step_two(Scalar4* d_vel,
                                Scalar3* d_accel,
                                unsigned int* d_group_members,
                                unsigned int group_size,
                                Scalar4* d_net_force,
                                unsigned int block_size,
                                Scalar deltaT,
                                Scalar exp_v_fac_thermo,
                                const GPUPartition& gpu_partition);

    }  // end namespace kernel
    }  // end namespace md
    }  // end namespace hoomd

#endif //__TWO_STEP_NVT_MTK_GPU_CUH__
