// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ActiveForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by ActiveForceComputeGPU.
*/

#ifndef __ACTIVE_FORCE_COMPUTE_GPU_CUH__
#define __ACTIVE_FORCE_COMPUTE_GPU_CUH__

cudaError_t gpu_compute_active_force_set_forces(const unsigned int group_size,
                                           unsigned int *d_rtag,
                                           unsigned int *d_groupTags,
                                           Scalar4 *d_force,
                                           Scalar4 *d_torque,
                                           Scalar4 *d_orientation,
                                           Scalar3 *d_f_actVec,
                                           Scalar *d_f_actMag,
                                           Scalar3 *d_t_actVec,
                                           Scalar *d_t_actMag,
                                           const Scalar3& P,
                                           Scalar rx,
                                           Scalar ry,
                                           Scalar rz,
                                           bool orientationLink,
                                           bool orientationReverseLink,
                                           const unsigned int N,
                                           unsigned int block_size);

cudaError_t gpu_compute_active_force_set_constraints(const unsigned int group_size,
                                                   unsigned int *d_rtag,
                                                   unsigned int *d_groupTags,
                                                   const Scalar4 *d_pos,
                                                   Scalar4 *d_force,
                                                   Scalar4 *d_torque,
                                                   Scalar3 *d_f_actVec,
                                                   Scalar3 *d_t_actVec,
                                                   const Scalar3& P,
                                                   Scalar rx,
                                                   Scalar ry,
                                                   Scalar rz,
                                                   unsigned int block_size);

cudaError_t gpu_compute_active_force_rotational_diffusion(const unsigned int group_size,
                                                       unsigned int *d_rtag,
                                                       unsigned int *d_groupTags,
                                                       const Scalar4 *d_pos,
                                                       Scalar4 *d_force,
                                                       Scalar4 *d_torque,
                                                       Scalar3 *d_f_actVec,
                                                       Scalar3 *d_t_actVec,
                                                       const Scalar3& P,
                                                       Scalar rx,
                                                       Scalar ry,
                                                       Scalar rz,
                                                       bool is2D,
                                                       const Scalar rotationDiff,
                                                       const unsigned int timestep,
                                                       const int seed,
                                                       unsigned int block_size);



#endif
