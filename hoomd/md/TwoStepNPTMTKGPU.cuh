// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __TWOSTEP_NPT_MTK_GPU_CUH__
#define __TWOSTEP_NPT_MTK_GPU_CUH__

#include <cuda_runtime.h>

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

/*! \file TwoStepNPTMTKGPU.cuh
    \brief Declares GPU kernel code for NPT integration on the GPU using the Martyna-Tobias-Klein (MTK) equations. Used by TwoStepNPTMTKGPU.
*/

//! Kernel driver for the the first step of the computation
cudaError_t gpu_npt_mtk_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar exp_thermo_fac,
                             Scalar *mat_exp_v,
                             Scalar *mat_exp_r,
                             Scalar *mat_exp_r_int,
                             Scalar deltaT,
                             bool rescale_all);

//! Kernel driver for wrapping particles back in the box (part of first step)
cudaError_t gpu_npt_mtk_wrap(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             const BoxDim& box);

//! Kernel driver for the the second step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_mtk_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             Scalar *mat_exp_v,
                             Scalar deltaT,
                             Scalar exp_thermo_fac);

//! Rescale all positions
void gpu_npt_mtk_rescale(unsigned int N,
                       Scalar4 *d_postype,
                       Scalar mat_exp_r_xx,
                       Scalar mat_exp_r_xy,
                       Scalar mat_exp_r_xz,
                       Scalar mat_exp_r_yy,
                       Scalar mat_exp_r_yz,
                       Scalar mat_exp_r_zz);

#endif
