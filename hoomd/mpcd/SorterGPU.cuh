// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_SORTER_GPU_CUH_
#define MPCD_SORTER_GPU_CUH_

/*!
 * \file mpcd/SorterGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SorterGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"

namespace mpcd
{
namespace gpu
{
cudaError_t sort_apply(Scalar4 *d_pos_alt,
                       Scalar4 *d_vel_alt,
                       unsigned int *d_tag_alt,
                       const Scalar4 *d_pos,
                       const Scalar4 *d_vel,
                       const unsigned int *d_tag,
                       const unsigned int *d_order,
                       const unsigned int N,
                       const unsigned int block_size);
} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_SORTER_GPU_CUH_
