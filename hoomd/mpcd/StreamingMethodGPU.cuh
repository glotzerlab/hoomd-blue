// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_STREAMING_METHOD_GPU_CUH_
#define MPCD_STREAMING_METHOD_GPU_CUH_

/*!
 * \file mpcd/StreamingMethodGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::StreamingMethodGPU
 */

#include <cuda_runtime.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace mpcd
{
namespace gpu
{

//! Kernel driver to stream particles ballistically
cudaError_t stream(Scalar4 *d_pos,
                   const Scalar4 *d_vel,
                   const BoxDim& box,
                   const Scalar dt,
                   const unsigned int N,
                   const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_STREAMING_METHOD_GPU_CUH_
