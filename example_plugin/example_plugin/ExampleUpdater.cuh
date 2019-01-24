// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXAMPLE_UPDATER_CUH_
#define _EXAMPLE_UPDATER_CUH_

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file ExampleUpdater.cuh
    \brief Declaration of CUDA kernels for ExampleUpdater
*/

// A C API call to run a CUDA kernel is needed for ExampleUpdaterGPU to call
//! Zeros velocities on the GPU
extern "C" cudaError_t gpu_zero_velocities(Scalar4 *d_vel, unsigned int N);

#endif // _EXAMPLE_UPDATER_CUH_
