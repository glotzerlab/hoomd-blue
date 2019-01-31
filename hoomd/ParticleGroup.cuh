// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "CachedAllocator.h"

/*! \file ParticleGroup.cuh
    \brief Contains GPU kernel code used by ParticleGroup
*/
#ifndef __PARTICLE_GROUP_CUH__
#define __PARTICLE_GROUP_CUH__

//! GPU method for rebuilding the index list of a ParticleGroup
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned int *d_is_member_tag,
                                   unsigned int *d_is_member,
                                   unsigned int *d_tag);

//! GPU method for compacting the group member indices
/*! \param N number of local particles
    \param d_is_member_tag Global lookup table for tag -> group membership
    \param d_is_member Array of membership flags
    \param d_member_idx Array of member indices
    \param d_tag Array of tags
    \param num_local_members Number of members on the local processor (return value)
*/
cudaError_t gpu_compact_index_list(unsigned int N,
                                   unsigned int *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int &num_local_members,
                                   unsigned int *d_tmp,
                                   CachedAllocator& alloc);
#endif
