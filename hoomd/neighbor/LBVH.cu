// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#include "LBVH.cuh"
#include "InsertOps.h"
#include "hoomd/extern/cub/cub/cub.cuh"

// macros for rounding HOOMD-blue Scalar to float for mixed precision
#ifdef SINGLE_PRECISION
#define __scalar2float_ru(x) (x)
#define __scalar2float_rd(x) (x)
#else
#define __scalar2float_ru(x) __double2float_ru((x))
#define __scalar2float_rd(x) __double2float_rd((x))
#endif

namespace neighbor
{
namespace gpu
{
namespace kernel
{
//! Compute the number of bits shared by Morton codes for primitives \a i and \a j.
/*!
 * \param d_codes List of Morton codes.
 * \param code_i Morton code corresponding to \a i.
 * \param i First primitive.
 * \param j Second primitive.
 * \param N Number of primitives.
 *
 * \returns Number of bits in longest common prefix or -1 if \a j lies outside [0,N).
 *
 * The longest common prefix of the Morton codes for \a i and \j is computed
 * using the __clz intrinsic. When \a i and \a j are the same, they share all 32
 * bits in the int representation of the Morton code. In that case, the common
 * prefix of \a i and \a j is used as a tie breaker.
 *
 * The user is required to supply \a code_i (even though it could also be looked
 * up from \a d_codes) for performance reasons, since code_i can be cached by
 * the caller if making multiple calls to ::delta for different \a j.
 */
__device__ __forceinline__ int delta(const unsigned int *d_codes,
                                     const unsigned int code_i,
                                     const int i,
                                     const int j,
                                     const unsigned int N)
    {
    if (j < 0 || j >= N)
        {
        return -1;
        }

    const unsigned int code_j = d_codes[j];

    if (code_i == code_j)
        {
        return (32 + __clz(i ^ j));
        }
    else
        {
        return __clz(code_i ^ code_j);
        }
    }

//! Kernel to generate the tree hierarchy
/*!
 * \param tree LBVH tree (raw pointers)
 * \param d_codes Sorted Morton codes for the primitives.
 * \param N Number of primitives
 *
 * One thread is used per *internal* node. (The LBVH guarantees that there are
 * exactly N-1 internal nodes.) The algorithm is given by Figure 4 of
 * <a href="https://dl.acm.org/citation.cfm?id=2383801">Karras</a>.
 */
__global__ void lbvh_gen_tree(LBVHData tree,
                              const unsigned int *d_codes,
                              const unsigned int N)
    {
    // one thread per internal node (= N-1 threads)
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N-1)
        return;

    const unsigned int code_i = d_codes[i];
    const int forward_prefix = delta(d_codes, code_i, i, i+1, N);
    const int backward_prefix = delta(d_codes, code_i, i, i-1, N);

    // get direction of the range based on sign
    const int d = (forward_prefix >= backward_prefix) - (forward_prefix < backward_prefix);

    // get minimum prefix
    const int min_prefix = delta(d_codes, code_i, i, i-d, N);

    // get maximum prefix by binary search
    int lmax = 2;
    while( delta(d_codes, code_i, i, i + d*lmax, N) > min_prefix)
        {
        lmax = lmax << 1;
        }
    int l = 0; int t = lmax;
    do
        {
        t = t >> 1;
        if (delta(d_codes, code_i, i, i + (l+t)*d, N) > min_prefix)
            l = l + t;
        }
    while (t > 1);
    const int j = i + l*d;

    // get the length of the common prefix
    const int common_prefix = delta(d_codes, code_i, i, j, N);

    // binary search to find split position
    int s = 0; t = l;
    do
        {
        t = (t + 1) >> 1;
        // if proposed split lies within range
        if (s+t < l)
            {
            const int split_prefix = delta(d_codes, code_i, i, i+(s+t)*d, N);

            // if new split shares a longer number of bits, accept it
            if (split_prefix > common_prefix)
                {
                s = s + t;
                }
            }
        }
    while (t > 1);
    const int split = i + s*d + min(d,0);

    const int left = (min(i,j) == split) ? split + (N-1) : split;
    const int right = (max(i,j) == (split + 1)) ? split + N : split + 1;

    // children
    tree.left[i] = left;
    tree.right[i] = right;

    // parents
    tree.parent[left] = i;
    tree.parent[right] = i;

    // root node (index 0) has no parent
    if (i == 0)
        {
        tree.parent[0] = LBVHSentinel;
        }
    }
} // end namespace kernel

/*!
 * \param d_tmp Temporary storage for CUB.
 * \param tmp_bytes Temporary storage size (B) for CUB.
 * \param d_codes Unsorted Morton codes.
 * \param d_sorted_codes Sorted Morton codes.
 * \param d_indexes Unsorted primitive indexes.
 * \param d_sorted_indexes Sorted primitive indexes.
 * \param N Number of primitives.
 * \param stream CUDA stream for kernel execution.
 *
 * \returns Two flags (swap) with the location of the sorted codes and indexes. If swap.x
 *          is 1, then the sorted codes are in \a d_codes and need to be swapped. Similarly,
 *          if swap.y is 1, then the sorted indexes are in \a d_indexes.
 *
 * The Morton codes are sorted in ascending order using radix sort in the CUB library.
 * This function must be called twice in order for the sort to occur. When \a d_tmp is NULL
 * on the first call, CUB sizes the temporary storage that is required and sets it in \a tmp_bytes.
 * Usually, this is a small amount and can be allocated from a buffer (e.g., a HOOMD-blue
 * CachedAllocator). Some versions of CUB were buggy and required \a d_tmp be allocated even
 * when \a tmp_bytes was 0. To bypass this, allocate a small amount (say, 4B) when \a tmp_bytes is 0.
 * The second call will then sort the Morton codes and indexes. The sorted data will be in the
 * appropriate buffer, which can be determined by the returned flags.
 */
uchar2 lbvh_sort_codes(void *d_tmp,
                       size_t &tmp_bytes,
                       unsigned int *d_codes,
                       unsigned int *d_sorted_codes,
                       unsigned int *d_indexes,
                       unsigned int *d_sorted_indexes,
                       const unsigned int N,
                       cudaStream_t stream)
    {

    cub::DoubleBuffer<unsigned int> d_keys(d_codes, d_sorted_codes);
    cub::DoubleBuffer<unsigned int> d_vals(d_indexes, d_sorted_indexes);

    cub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, sizeof(unsigned int)*8, stream);

    uchar2 swap = make_uchar2(0,0);
    if (d_tmp != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the sorted array (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

/*!
 * \param tree LBVH tree (raw pointers).
 * \param d_codes Sorted Morton codes for the primitives.
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \sa kernel::lbvh_gen_tree
 */
void lbvh_gen_tree(const LBVHData tree,
                   const unsigned int *d_codes,
                   const unsigned int N,
                   const unsigned int block_size,
                   cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_gen_tree);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = ((N-1) + run_block_size - 1)/run_block_size;
    kernel::lbvh_gen_tree<<<num_blocks, run_block_size, 0, stream>>>(tree, d_codes, N);
    }

//! Template declarations for lbvh_gen_codes
template void lbvh_gen_codes(unsigned int *d_codes,
                             unsigned int *d_indexes,
                             const PointInsertOp& insert,
                             const Scalar3 lo,
                             const Scalar3 hi,
                             const unsigned int N,
                             const unsigned int block_size,
                             cudaStream_t stream);

template void lbvh_gen_codes(unsigned int *d_codes,
                             unsigned int *d_indexes,
                             const SphereInsertOp& insert,
                             const Scalar3 lo,
                             const Scalar3 hi,
                             const unsigned int N,
                             const unsigned int block_size,
                             cudaStream_t stream);

//! Template declarations for lbvh_bubble_aabbs
template void lbvh_bubble_aabbs(const LBVHData tree,
                                const PointInsertOp& insert,
                                unsigned int *d_locks,
                                const unsigned int N,
                                const unsigned int block_size,
                                cudaStream_t stream);

template void lbvh_bubble_aabbs(const LBVHData tree,
                                const SphereInsertOp& insert,
                                unsigned int *d_locks,
                                const unsigned int N,
                                const unsigned int block_size,
                                cudaStream_t stream);

template void lbvh_one_primitive(const LBVHData tree,
                                 const PointInsertOp& insert,
                                 cudaStream_t stream);
template void lbvh_one_primitive(const LBVHData tree,
                                 const SphereInsertOp& insert,
                                 cudaStream_t stream);

} // end namespace gpu
} // end namespace neighbor
