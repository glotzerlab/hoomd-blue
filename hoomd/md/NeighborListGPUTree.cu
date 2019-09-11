// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "NeighborListGPUTree.cuh"

#include "hoomd/neighbor/LBVH.cuh"
#include "hoomd/neighbor/LBVHTraverser.cuh"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#include "hoomd/extern/cub/cub/cub.cuh"

__global__ void gpu_nlist_mark_types_kernel(unsigned int *d_types,
                                            unsigned int *d_indexes,
                                            unsigned int *d_lbvh_errors,
                                            Scalar4 *d_last_pos,
                                            const Scalar4 *d_pos,
                                            const unsigned int N,
                                            const unsigned int nghosts,
                                            const BoxDim box,
                                            const Scalar3 ghost_width)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N+nghosts)
        return;

    // acquire particle data
    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos, ghost_width);

    /*
     * check if the particle is inside the unit cell + ghost layer.
     * we silently ignore ghosts outside of this width, since they could be in bonds, etc.
     */
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001)) ||
        (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001)) ||
        (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
        {
        // error for owned particle
        if (idx < N)
            {
            atomicMax(d_lbvh_errors,idx+1);
            return;
            }
        else // silent for ghosts
            {
            type = NeighborListTypeSentinel;
            }
        }

    d_types[idx] = type;
    d_indexes[idx] = idx;
    // record as "last" position of owned particles
    if (idx < N)
        {
        d_last_pos[idx] = postype;
        }
    }

cudaError_t gpu_nlist_mark_types(unsigned int *d_types,
                                 unsigned int *d_indexes,
                                 unsigned int *d_lbvh_errors,
                                 Scalar4 *d_last_pos,
                                 const Scalar4 *d_pos,
                                 const unsigned int N,
                                 const unsigned int nghosts,
                                 const BoxDim& box,
                                 const Scalar3 ghost_width,
                                 const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_mark_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    const unsigned int run_block_size = min(block_size,max_block_size);
    const unsigned int num_blocks = ((N+nghosts) + run_block_size - 1)/run_block_size;
    gpu_nlist_mark_types_kernel<<<num_blocks, run_block_size>>>(d_types,
                                                                d_indexes,
                                                                d_lbvh_errors,
                                                                d_last_pos,
                                                                d_pos,
                                                                N,
                                                                nghosts,
                                                                box,
                                                                ghost_width);
    return cudaSuccess;
    }

uchar2 gpu_nlist_sort_types(void *d_tmp,
                            size_t &tmp_bytes,
                            unsigned int *d_types,
                            unsigned int *d_sorted_types,
                            unsigned int *d_indexes,
                            unsigned int *d_sorted_indexes,
                            const unsigned int N,
                            const unsigned int num_bits)
    {
    cub::DoubleBuffer<unsigned int> d_keys(d_types, d_sorted_types);
    cub::DoubleBuffer<unsigned int> d_vals(d_indexes, d_sorted_indexes);

    // we counted number of bits to sort, so the range of bit indexes is [0,num_bits)
    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, num_bits);

    uchar2 swap = make_uchar2(0,0);
    if (d_tmp != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the sorted array (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

__global__ void gpu_nlist_count_types_kernel(unsigned int *d_first,
                                             unsigned int *d_last,
                                             const unsigned int *d_types,
                                             const unsigned int ntypes,
                                             const unsigned int N)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    // my type
    const unsigned int type = d_types[idx];
    // look to left if not first
    const unsigned int left = (idx > 0) ? d_types[idx-1] : NeighborListTypeSentinel;
    // look to right if not last
    const unsigned int right = (idx < N-1) ? d_types[idx+1] : NeighborListTypeSentinel;

    // if left is not same as self (or idx == 0 by use of sentinel), this is the first index in the type
    if (left != type && type < ntypes)
        {
        d_first[type] = idx;
        }
    // if right is not the same as self (or idx == N-1 by use of sentinel), this is the last index in the type
    if (right != type && type < ntypes)
        {
        d_last[type] = idx + 1;
        }
    }

cudaError_t gpu_nlist_count_types(unsigned int *d_first,
                                  unsigned int *d_last,
                                  const unsigned int *d_types,
                                  const unsigned int ntypes,
                                  const unsigned int N,
                                  const unsigned int block_size)

    {
    // initially, fill all cells as empty
    thrust::fill(thrust::device, d_first, d_first+ntypes, NeighborListTypeSentinel);
    cudaMemset(d_last, 0, sizeof(unsigned int)*ntypes);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_count_types_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    gpu_nlist_count_types_kernel<<<N/run_block_size + 1, run_block_size>>>(d_first,
                                                                           d_last,
                                                                           d_types,
                                                                           ntypes,
                                                                           N);
    return cudaSuccess;
    }

__global__ void gpu_nlist_copy_primitives_kernel(unsigned int *d_traverse_order,
                                                 const unsigned int *d_indexes,
                                                 const unsigned int *d_primitives,
                                                 const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    const unsigned int primitive = d_primitives[idx];
    d_traverse_order[idx] = __ldg(d_indexes + primitive);
    }

cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_nlist_copy_primitives_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    int run_block_size = min(block_size,max_block_size);
    gpu_nlist_copy_primitives_kernel<<<N/run_block_size + 1, run_block_size>>>(d_traverse_order,
                                                                               d_indexes,
                                                                               d_primitives,
                                                                               N);
    return cudaSuccess;
    }

// explicit templates for neighbor::LBVH with PointMapInsertOp and NullOp
template void neighbor::gpu::lbvh_gen_codes(unsigned int *, unsigned int *, const PointMapInsertOp&,
    const Scalar3, const Scalar3, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_bubble_aabbs(const neighbor::gpu::LBVHData, const PointMapInsertOp&,
    unsigned int *, const unsigned int, const unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_one_primitive(const neighbor::gpu::LBVHData, const PointMapInsertOp&, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
    const ParticleQueryOp<false,false>&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
    const ParticleQueryOp<false,true>&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
    const ParticleQueryOp<true,false>&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);
template void neighbor::gpu::lbvh_traverse_ropes(NeighborListOp&, const neighbor::gpu::LBVHCompressedData&,
    const ParticleQueryOp<true,true>&, const Scalar3 *, unsigned int, unsigned int, cudaStream_t);
