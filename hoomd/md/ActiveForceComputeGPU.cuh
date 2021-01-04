// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ActiveForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveForceComputeGPU.
*/

#ifndef __ACTIVE_FORCE_COMPUTE_GPU_CUH__
#define __ACTIVE_FORCE_COMPUTE_GPU_CUH__

hipError_t gpu_compute_active_force_set_forces(const unsigned int group_size,
                                           unsigned int *d_index_array,
                                           Scalar4 *d_force,
                                           Scalar4 *d_torque,
                                           const Scalar4 *d_pos,
                                           const Scalar4 *d_orientation,
                                           const Scalar4 *d_f_act,
                                           const Scalar4 *d_t_act,
                                           const Scalar3& P,
                                           const Scalar rx,
                                           const Scalar ry,
                                           const Scalar rz,
                                           const unsigned int N,
                                           unsigned int block_size);

hipError_t gpu_compute_active_force_set_constraints(const unsigned int group_size,
                                                   unsigned int *d_index_array,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_orientation,
                                                   const Scalar4 *d_f_act,
                                                   const Scalar3& P,
                                                   const Scalar rx,
                                                   const Scalar ry,
                                                   const Scalar rz,
                                                   unsigned int block_size);

hipError_t gpu_compute_active_force_rotational_diffusion(const unsigned int group_size,
                                                       unsigned int *d_tag,
                                                       unsigned int *d_index_array,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_orientation,
                                                       const Scalar4 *d_f_act,
                                                       const Scalar3& P,
                                                       const Scalar rx,
                                                       const Scalar ry,
                                                       const Scalar rz,
                                                       bool is2D,
                                                       const Scalar rotationDiff,
                                                       const unsigned int timestep,
                                                       const int seed,
                                                       unsigned int block_size);



#endif
