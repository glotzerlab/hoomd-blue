// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_CUH_
#define NEIGHBOR_LBVH_CUH_

#include "hoomd/HOOMDMath.h"

#include "BoundingVolumes.h"

namespace neighbor
{
namespace gpu
{
// LBVH sentinel has value of max signed int (~2 billion)
const int LBVHSentinel=0x7fffffff;

//! Linear bounding volume hierarchy raw data
/*!
 * LBVHData is a lightweight struct representation of the LBVH. It is useful for passing tree data
 * to a CUDA kernel. It is valid to set a pointer to NULL if it is not required, but the caller
 * for doing so responsibly.
 */
struct LBVHData
    {
    int* parent;                        //!< Parent node
    int* left;                          //!< Left child
    int* right;                         //!< Right child
    const unsigned int* primitive;      //!< Primitives
    float3* lo;                         //!< Lower bound of AABB
    float3* hi;                         //!< Upper bound of AABB
    int root;                           //!< Root index
    };

//! Generate the Morton codes
template<class InsertOpT>
void lbvh_gen_codes(unsigned int *d_codes,
                    unsigned int *d_indexes,
                    const InsertOpT& insert,
                    const Scalar3 lo,
                    const Scalar3 hi,
                    const unsigned int N,
                    const unsigned int block_size,
                    cudaStream_t stream = 0);

//! Sort the Morton codes.
uchar2 lbvh_sort_codes(void *d_tmp,
                       size_t &tmp_bytes,
                       unsigned int *d_codes,
                       unsigned int *d_sorted_codes,
                       unsigned int *d_indexes,
                       unsigned int *d_sorted_indexes,
                       const unsigned int N,
                       cudaStream_t stream = 0);

//! Generate the tree hierarchy
void lbvh_gen_tree(const LBVHData tree,
                   const unsigned int *d_codes,
                   const unsigned int N,
                   const unsigned int block_size,
                   cudaStream_t stream = 0);

//! Bubble the bounding boxes up the tree hierarchy.
template<class InsertOpT>
void lbvh_bubble_aabbs(const LBVHData tree,
                       const InsertOpT& insert,
                       unsigned int *d_locks,
                       const unsigned int N,
                       const unsigned int block_size,
                       cudaStream_t stream = 0);

template<class InsertOpT>
void lbvh_one_primitive(const LBVHData tree,
                        const InsertOpT& insert,
                        cudaStream_t stream = 0);

#ifdef NVCC
namespace kernel
{
//! Expand a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
/*!
 * \param v unsigned integer with 10 bits set
 * \returns The integer expanded with two zeros interleaved between bits
 * http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
 */
__device__ __forceinline__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

//! Compute the 30-bit Morton code for a tuple of binned indexes.
/*!
 * \param point (x,y,z) tuple of bin indexes.
 * \returns 30-bit Morton code corresponding to \a point.
 *
 * The Morton code is formed by first expanding the bits of each component (see ::expandBits),
 * and then bitshifting to interleave them. The Morton code then has a representation::
 *
 *  x0y0z0x1y1z1...
 *
 * where indices refer to the bitwise representation of each component.
 */
__device__ __forceinline__ unsigned int calcMortonCode(uint3 point)
    {
    return 4 * expandBits(point.x) + 2 * expandBits(point.y) + expandBits(point.z);
    }

//! Convert a fraction to [0,1023]
/*
 * \param f Fractional coordinate lying in [0,1].
 * \returns Bin integer lying in [0,1023]
 *
 * The range of the binned integer corresponds to the maximum value that can be
 * stored in a 10-bit integer. When \a f lies outside [0,1], the bin is clamped to
 * the ends of the range.
 */
__device__ __forceinline__ unsigned int fractionToBin(float f)
    {
    return static_cast<unsigned int>(fminf(fmaxf(f * 1023.f, 0.f), 1023.f));
    }

//! Kernel to generate the Morton codes
/*!
 * \param d_codes Generated Morton codes.
 * \param d_indexes Generated index for the primitive.
 * \param insert Insert operation.
 * \param lo Lower bound of scene.
 * \param hi Upper bound of scene.
 * \param N Number of primitives.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * One thread is used to process each primitive. The point is binned into
 * one of 2^10 bins using its fractional coordinate between \a lo and \a hi.
 * The bins are converted to a Morton code. The Morton code and corresponding
 * primitive index are stored. The reason for storing the primitive index now
 * is for subsequent sorting (see ::lbvh_sort_codes).
 */
template<class InsertOpT>
__global__ void lbvh_gen_codes(unsigned int *d_codes,
                               unsigned int *d_indexes,
                               const InsertOpT insert,
                               const Scalar3 lo,
                               const Scalar3 hi,
                               const unsigned int N)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    // real space coordinate of aabb center
    const BoundingBox b = insert.get(idx);
    const float3 r = b.getCenter();

    // fractional coordinate
    const float3 f = make_float3((r.x - lo.x) / (hi.x - lo.x),
                                 (r.y - lo.y) / (hi.y - lo.y),
                                 (r.z - lo.z) / (hi.z - lo.z));

    // bin fractional coordinate
    const uint3 q = make_uint3(fractionToBin(f.x), fractionToBin(f.y), fractionToBin(f.z));

    // compute morton code
    const unsigned int code = calcMortonCode(q);

    // write out morton code and primitive index
    d_codes[idx] = code;
    d_indexes[idx] = idx;
    }

//! Bubble the bounding boxes up the tree hierarchy.
/*!
 * \param tree LBVH tree (raw pointers).
 * \param insert the insert operation to get the aabbs
 * \param d_locks Temporary storage for state of internal nodes.
 * \param N Number of primitives.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * One thread originally processes each primitive. In order to support mixed precision,
 * the Scalar4 representation of the primitive is converted to two float3s that define
 * the lower and upper bounds using CUDA intrinsics to round down or up. (If Scalar = float,
 * then these bounds are equal.) This bounding box is stored for the leaf. Then, each thread
 * begins to process up the tree hierarchy.
 *
 * The second thread to reach each node processes the node, which ensures that all children
 * have already been processed. The order to reach the node is determined by an atomic
 * operation on \a d_locks. The bounding box of the node being processed is determined by
 * merging the bounding box of the child processing its parent with the bounding box of its
 * sibling. The process is then repeated until the root node is reached.
 *
 * The InsertOpT is used to determine initial bounding boxes for the primitives.
 *
 * \note
 * A __threadfence() is employed after the AABB is stored to ensure that it is visible to
 * other threads reading from global memory.
 */
template<class InsertOpT>
__global__ void lbvh_bubble_aabbs(const LBVHData tree,
                                  const InsertOpT insert,
                                  unsigned int *d_locks,
                                  const unsigned int N)
    {
    // one thread per point
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    // determine lower and upper bounds of the primitive, even in mixed precision
    BoundingBox b = insert.get(tree.primitive[idx]);
    float3 lo = b.lo;
    float3 hi = b.hi;

    // set aabb for the leaf node
    int last = N-1+idx;
    tree.lo[last] = lo;
    tree.hi[last] = hi;
    __threadfence();

    int current = tree.parent[last];
    while (current != LBVHSentinel)
        {
        // parent is processed by the second thread to reach it
        unsigned int lock = atomicAdd(d_locks + current, 1);
        if (!lock)
            return;

        // look for the sibling of the current thread with speculation
        int sibling = tree.left[current];
        if (sibling == last)
            {
            sibling = tree.right[current];
            }

        // compute min / max bounds of the current thread with its sibling
        const float3 sib_lo = tree.lo[sibling];
        if (sib_lo.x < lo.x) lo.x = sib_lo.x;
        if (sib_lo.y < lo.y) lo.y = sib_lo.y;
        if (sib_lo.z < lo.z) lo.z = sib_lo.z;

        const float3 sib_hi = tree.hi[sibling];
        if (sib_hi.x > hi.x) hi.x = sib_hi.x;
        if (sib_hi.y > hi.y) hi.y = sib_hi.y;
        if (sib_hi.z > hi.z) hi.z = sib_hi.z;

        // write out bounding box to global memory
        tree.lo[current] = lo;
        tree.hi[current] = hi;
        __threadfence();

        // move up tree
        last = current;
        current = tree.parent[current];
        }
    }

template<class InsertOpT>
__global__ void lbvh_one_primitive(const LBVHData tree,
                                   const InsertOpT insert)
    {
    // one thread only
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= 1)
        return;

    const BoundingBox b = insert.get(0);

    tree.parent[0] = LBVHSentinel;
    tree.lo[0] = b.lo;
    tree.hi[0] = b.hi;
    }

} // end namespace kernel

/*!
 * \param d_codes Generated Morton codes.
 * \param d_indexes Generated index for the primitive.
 * \param insert Insert operation.
 * \param lo Lower bound of scene.
 * \param hi Upper bound of scene.
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * \sa kernel::lbvh_gen_codes
 */
template<class InsertOpT>
void lbvh_gen_codes(unsigned int *d_codes,
                    unsigned int *d_indexes,
                    const InsertOpT& insert,
                    const Scalar3 lo,
                    const Scalar3 hi,
                    const unsigned int N,
                    const unsigned int block_size,
                    cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_gen_codes<InsertOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;
    kernel::lbvh_gen_codes<<<num_blocks, run_block_size, 0, stream>>>(d_codes, d_indexes, insert, lo, hi, N);
    }

/*!
 * \param tree LBVH tree (raw pointers).
 * \param d_locks Temporary storage for state of internal nodes.
 * \param insert The insert operation to obtain the aabbs
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * \sa kernel::lbvh_bubble_aabbs
 *
 * \a d_locks is overwritten before the kernel is launched.
 */
template<class InsertOpT>
void lbvh_bubble_aabbs(const LBVHData tree,
                       const InsertOpT& insert,
                       unsigned int *d_locks,
                       const unsigned int N,
                       const unsigned int block_size,
                       cudaStream_t stream)
    {
    cudaMemset(d_locks, 0, (N-1)*sizeof(unsigned int));

    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_bubble_aabbs<InsertOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N + run_block_size - 1)/run_block_size;
    kernel::lbvh_bubble_aabbs<<<num_blocks, run_block_size, 0, stream>>>(tree, insert, d_locks, N);
    }

template<class InsertOpT>
void lbvh_one_primitive(const LBVHData tree,
                        const InsertOpT& insert,
                        cudaStream_t stream)
    {
    kernel::lbvh_one_primitive<<<1, 1, 0, stream>>>(tree, insert);
    }
#endif // NVCC

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_CUH_
