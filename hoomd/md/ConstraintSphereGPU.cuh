// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ConstraintSphereGPU.cuh
    \brief Declares GPU kernel code for calculating sphere constraint forces on the GPU. Used by ConstraintSphereGPU.
*/

#ifndef __CONSTRAINT_SPHERE_GPU_CUH__
#define __CONSTRAINT_SPHERE_GPU_CUH__

//! Kernel driver that computes harmonic bond forces for HarmonicBondForceComputeGPU
cudaError_t gpu_compute_constraint_sphere_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const Scalar4 *d_vel,
                                                 const Scalar4 *d_net_force,
                                                 const Scalar3& P,
                                                 Scalar r,
                                                 Scalar deltaT,
                                                 unsigned int block_size);

#endif
