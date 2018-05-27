// Copyright (c) 2009-2018 The Regents of the University of Michigan
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
                                   unsigned int *d_member_idx,
                                   unsigned int *d_tag,
                                   unsigned int &num_local_members,
                                   unsigned int *d_tmp,
                                   const CachedAllocator& alloc);
#endif
