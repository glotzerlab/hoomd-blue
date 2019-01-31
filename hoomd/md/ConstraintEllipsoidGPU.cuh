// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ConstraintEllipsoidGPU.cuh
    \brief Declares GPU kernel code for calculating sphere constraint forces on the GPU. Used by ConstraintEllipsoidGPU.
*/

#ifndef __CONSTRAINT_ELLIPSOID_GPU_CUH__
#define __CONSTRAINT_ELLIPSOID_GPU_CUH__

//! Kernel driver that computes harmonic bond forces for HarmonicBondForceComputeGPU
cudaError_t gpu_compute_constraint_ellipsoid_constraint(const unsigned int *d_group_members,
                                                 unsigned int group_size,
                                                 const unsigned int N,
                                                 Scalar4 *d_pos,
                                                 const Scalar3 P,
                                                 Scalar rx,
                                                 Scalar ry,
                                                 Scalar rz,
                                                 unsigned int block_size);

#endif
