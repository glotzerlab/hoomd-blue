// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_SORTER_GPU_CUH_
#define MPCD_SORTER_GPU_CUH_

/*!
 * \file mpcd/SorterGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SorterGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Kernel driver to apply sorted particle order
cudaError_t sort_apply(Scalar4* d_pos_alt,
                       Scalar4* d_vel_alt,
                       unsigned int* d_tag_alt,
                       const Scalar4* d_pos,
                       const Scalar4* d_vel,
                       const unsigned int* d_tag,
                       const unsigned int* d_order,
                       const unsigned int N,
                       const unsigned int block_size);

//! Kernel driver to sort order
cudaError_t set_order(unsigned int* d_order,
                      unsigned int* d_cell_id,
                      const Scalar4* d_vel,
                      const unsigned int N_mpcd,
                      const unsigned int block_size);

cudaError_t
compute_order(unsigned int* d_order, unsigned int* d_cell_id, const unsigned int N_mpcd);

//! Driver for thrust to perform cell-list stream compaction
unsigned int sort_cell_compact(unsigned int* d_order,
                               const unsigned int* d_cell_list,
                               const unsigned int num_items,
                               const unsigned int N);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SORTER_GPU_CUH_
