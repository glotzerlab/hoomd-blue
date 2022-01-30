// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshGroupData.cuh
    \brief Defines the helper functions (GPU version) for updating the GPU bonded group tables
 */

#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#include "HOOMDMath.h"

#ifndef __MESH_GROUP_DATA_CUH__
#define __MESH_GROUP_DATA_CUH__

namespace hoomd
    {
template<unsigned int group_size, typename group_t>
void gpu_update_mesh_table(const unsigned int n_groups,
                           const unsigned int N,
                           const group_t* d_group_table,
                           const typeval_union* d_group_typeval,
                           const unsigned int* d_rtag,
                           unsigned int* d_n_groups,
                           unsigned int max_n_groups,
                           unsigned int* d_condition,
                           unsigned int next_flag,
                           unsigned int& flag,
                           group_t* d_pidx_group_table,
                           unsigned int* d_pidx_gpos_table,
                           const unsigned int pidx_group_table_pitch,
                           unsigned int* d_scratch_g,
                           unsigned int* d_scratch_idx,
                           unsigned int* d_offsets,
                           CachedAllocator& alloc);
    }  // end namespace hoomd
#endif // __MESH_GROUP_DATA_CUH__
