// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ParticleData.cuh"

#include "hoomd/extern/kernels/scan.cuh"

/*! \file ParticleGroup.cu
    \brief Contains GPU kernel code used by ParticleGroup
*/

//! GPU kernel to translate between global and local membership lookup table
__global__ void gpu_rebuild_index_list_kernel(unsigned int N,
                                              unsigned int *d_tag,
                                              unsigned char *d_is_member_tag,
                                              unsigned char *d_is_member)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    unsigned int tag = d_tag[idx];

    d_is_member[idx] = d_is_member_tag[tag];
    }

__global__ void gpu_scatter_member_indices(unsigned int N,
    const unsigned int *d_scan,
    const unsigned char *d_is_member,
    unsigned *d_member_idx)
    {
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx >= N) return;

    if (d_is_member[idx])
        d_member_idx[d_scan[idx]] = idx;
    }

//! GPU method for rebuilding the index list of a ParticleGroup
/*! \param N number of local particles
    \param d_is_member_tag Global lookup table for tag -> group membership
    \param d_is_member Array of membership flags
    \param d_member_idx Array of member indices
    \param d_tag Array of tags
    \param num_local_members Number of members on the local processor (return value)
*/
cudaError_t gpu_rebuild_index_list(unsigned int N,
                                   unsigned char *d_is_member_tag,
                                   unsigned char *d_is_member,
                                   unsigned int *d_member_idx,
                                   unsigned int *d_tag,
                                   unsigned int &num_local_members,
                                   unsigned int *d_tmp,
                                   mgpu::ContextPtr mgpu_context)
    {
    assert(d_is_member);
    assert(d_is_member_tag);
    assert(d_member_idx);
    assert(d_tag);

    unsigned int block_size = 512;
    unsigned int n_blocks = N/block_size + 1;

    gpu_rebuild_index_list_kernel<<<n_blocks,block_size>>>(N,
                                                         d_tag,
                                                         d_is_member_tag,
                                                         d_is_member);

    // compute member_idx offsets
    mgpu::Scan<mgpu::MgpuScanTypeExc>(d_is_member, N, (unsigned int) 0, mgpu::plus<unsigned int>(),
        (unsigned int *) NULL, &num_local_members, d_tmp, *mgpu_context);

    // fill member_idx array
    gpu_scatter_member_indices<<<n_blocks, block_size>>>(N, d_tmp, d_is_member, d_member_idx);

    return cudaSuccess;
    }
