// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_CELL_LIST_GPU_CUH_
#define MPCD_CELL_LIST_GPU_CUH_

/*!
 * \file mpcd/CellListGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellListGPU
 */

#include <cuda_runtime.h>

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace mpcd
{
namespace gpu
{

//! Kernel driver to compute mpcd cell list
cudaError_t compute_cell_list(unsigned int *d_cell_np,
                              unsigned int *d_cell_list,
                              uint3 *d_conditions,
                              Scalar4 *d_vel,
                              unsigned int *d_embed_cell_ids,
                              const Scalar4 *d_pos,
                              const Scalar4 *d_pos_embed,
                              const unsigned int *d_embed_member_idx,
                              const uchar3& periodic,
                              const int3& origin_idx,
                              const Scalar3& grid_shift,
                              const Scalar3& global_lo,
                              const uint3& n_global_cell,
                              const Scalar cell_size,
                              const unsigned int cell_np_max,
                              const Index3D& cell_indexer,
                              const Index2D& cell_list_indexer,
                              const unsigned int N_mpcd,
                              const unsigned int N_tot,
                              const unsigned int block_size);

//! Kernel driver to check if any embedded particles require migration
cudaError_t cell_check_migrate_embed(unsigned int *d_migrate_flag,
                                     const Scalar4 *d_pos,
                                     const unsigned int *d_group,
                                     const BoxDim& box,
                                     const unsigned int num_dim,
                                     const unsigned int N,
                                     const unsigned int block_size);

//! Kernel drive to apply sorted order to MPCD particles in cell list
cudaError_t cell_apply_sort(unsigned int *d_cell_list,
                            const unsigned int *d_rorder,
                            const unsigned int *d_cell_np,
                            const Index2D& cli,
                            const unsigned int N_mpcd,
                            const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_CELL_LIST_GPU_CUH_
