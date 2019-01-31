// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"


#ifndef _BERENDSEN_GPU_CUH_
#define _BERENDSEN_GPU_CUH_

/*! \file TwoStepBerendsenGPU.cuh
    \brief Declaration of CUDA kernels for Berendsen thermostat on the GPU
*/

//! Kernel driver for gpu_berendsen_step_one_kernel
cudaError_t gpu_berendsen_step_one(Scalar4 *d_pos,
                                   Scalar4 *d_vel,
                                   const Scalar3 *d_accel,
                                   int3 *d_image,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const BoxDim& box,
                                   unsigned int block_size,
                                   Scalar lambda,
                                   Scalar deltaT);

//! Kernel driver for gpu_berendsen_step_two_kernel
cudaError_t gpu_berendsen_step_two(Scalar4 *d_vel,
                                   Scalar3 *d_accel,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   Scalar4 *d_net_force,
                                   unsigned int block_size,
                                   Scalar deltaT);

#endif // _BERENDSEN_GPU_CUH_
