// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Enforce2DUpdaterGPU.cuh
    \brief Declares GPU kernel code for constraining particles to the xy plane on the
            GPU. Used by Enforce2DUpdaterGPU.
*/

#ifndef __ENFORCE2DUPDATER_CUH__
#define __ENFORCE2DUPDATER_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

//! Kernel driver for the enforce 2D update called by Enforce2DUpdaterGPU
cudaError_t gpu_enforce2d(const unsigned int N,
                          Scalar4 *d_vel,
                          Scalar3 *d_accel);

#endif
