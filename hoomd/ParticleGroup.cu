// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ParticleData.cuh"
#include "ParticleGroup.cuh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hipcub/hipcub.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#pragma GCC diagnostic pop

/*! \file ParticleGroup.cu
    \brief Contains GPU kernel code used by ParticleGroup
*/

namespace hoomd
    {
namespace kernel
    {
//! GPU kernel to translate between global and local membership lookup table
__global__ void gpu_rebuild_index_list_kernel(unsigned int N,
                                              unsigned int* d_tag,
                                              unsigned int* d_is_member_tag,
                                              unsigned int* d_is_member)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    unsigned int tag = d_tag[idx];

    d_is_member[idx] = d_is_member_tag[tag];
    }

__global__ void gpu_scatter_member_indices(unsigned int N,
                                           const unsigned int* d_scan,
                                           const unsigned int* d_is_member,
                                           unsigned* d_member_idx)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

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
hipError_t gpu_rebuild_index_list(unsigned int N,
                                  unsigned int* d_is_member_tag,
                                  unsigned int* d_is_member,
                                  unsigned int* d_tag)
    {
    assert(d_is_member);
    assert(d_is_member_tag);
    assert(d_tag);

    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    hipLaunchKernelGGL(gpu_rebuild_index_list_kernel,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_tag,
                       d_is_member_tag,
                       d_is_member);
    return hipSuccess;
    }

//! GPU method for compacting the group member indices
/*! \param N number of local particles
    \param d_is_member_tag Global lookup table for tag -> group membership
    \param d_is_member Array of membership flags
    \param d_member_idx Array of member indices
    \param d_tag Array of tags
    \param num_local_members Number of members on the local processor (return value)
*/
hipError_t gpu_compact_index_list(unsigned int N,
                                  unsigned int* d_is_member,
                                  unsigned int* d_member_idx,
                                  unsigned int& num_local_members,
                                  unsigned int* d_tmp,
                                  CachedAllocator& alloc)
    {
    assert(d_is_member);
    assert(d_member_idx);

    // compute member_idx offsets
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // determine size of temporary storage
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_is_member, d_tmp, N);

    d_temp_storage = alloc.getTemporaryBuffer<char>(temp_storage_bytes);
    hipcub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_is_member, d_tmp, N);
    alloc.deallocate((char*)d_temp_storage);

    thrust::device_ptr<unsigned int> is_member(d_is_member);
#ifdef __HIP_PLATFORM_HCC__
    num_local_members = thrust::reduce(thrust::hip::par(alloc),
#else
    num_local_members = thrust::reduce(thrust::cuda::par(alloc),
#endif
                                       is_member,
                                       is_member + N);

    // fill member_idx array
    unsigned int block_size = 256;
    unsigned int n_blocks = N / block_size + 1;

    hipLaunchKernelGGL(gpu_scatter_member_indices,
                       dim3(n_blocks),
                       dim3(block_size),
                       0,
                       0,
                       N,
                       d_tmp,
                       d_is_member,
                       d_member_idx);

    return hipSuccess;
    }

    } // end namespace kernel

    } // end namespace hoomd
