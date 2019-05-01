// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file BondedGroupData.cuh
    \brief Defines the helper functions (GPU version) for updating the GPU bonded group tables
 */

#include "hoomd/extern/util/mgpucontext.h"

#include "HOOMDMath.h"

#ifndef __BONDED_GROUP_DATA_CUH__
#define __BONDED_GROUP_DATA_CUH__

#ifdef NVCC
//! Sentinel value
const unsigned int GROUP_NOT_LOCAL = 0xffffffff;

//! Storage for group members (GPU declaration)
template<unsigned int group_size>
union group_storage
    {
    unsigned int tag[group_size]; // access 'tags'
    unsigned int idx[group_size]; // access 'indices'
    };

//! A union to allow storing a Scalar constraint value or a type integer (GPU declaration)
union typeval_union
    {
    unsigned int type;
    Scalar val;
    };

//! Packed group entry for communication (GPU declaration)
template<unsigned int group_size>
struct packed_storage
    {
    group_storage<group_size> tags;  //!< Member tags
    typeval_union typeval;               //!< Type or constraint value of bonded group
    unsigned int group_tag;          //!< Tag of this group
    group_storage<group_size> ranks; //!< Current list of member ranks
    };

#else
//! Forward declaration of group_storage
template<unsigned int group_size>
union group_storage;

//! Forward declaration of packed_storage
template<unsigned int group_size>
struct packed_storage;

//! Forward declaration of typeval union
union typeval_union;
#endif

template<unsigned int group_size, typename group_t>
void gpu_update_group_table(
    const unsigned int n_groups,
    const unsigned int N,
    const group_t* d_group_table,
    const typeval_union *d_group_typeval,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_t *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    bool has_type_mapping,
    mgpu::ContextPtr mgpu_context
    );
#endif // __BONDED_GROUP_DATA_CUH__
