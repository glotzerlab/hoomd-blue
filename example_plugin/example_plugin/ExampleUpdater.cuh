// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _EXAMPLE_UPDATER_CUH_
#define _EXAMPLE_UPDATER_CUH_

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file ExampleUpdater.cuh
    \brief Declaration of CUDA kernels for ExampleUpdater
*/

namespace hoomd
    {
namespace kernel
    {
//! Zeros velocities on the GPU
hipError_t gpu_zero_velocities(Scalar4* d_vel, unsigned int N);

    } // end namespace kernel
    } // end namespace hoomd

#endif // _EXAMPLE_UPDATER_CUH_
