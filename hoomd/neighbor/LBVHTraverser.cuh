// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_TRAVERSER_CUH_
#define NEIGHBOR_LBVH_TRAVERSER_CUH_

#include "hoomd/HOOMDMath.h"
#include "LBVH.cuh"
#include "BoundingVolumes.h"

namespace neighbor
{
namespace gpu
{

//! Lightweight data structure to hold the compressed LBVH.
struct LBVHCompressedData
    {
    int root;       //!< Root index of the LBVH
    int4* data;     //!< Compressed LBVH data.
    float3* lo;     //!< Lower bound used in compression.
    float3* hi;     //!< Upper bound used in compression.
    float3* bins;   //!< Bin spacing used in compression.
    };

//! Compress LBVH for rope traversal.
template<class TransformOpT>
void lbvh_compress_ropes(LBVHCompressedData ctree,
                         const TransformOpT& transform,
                         const LBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size,
                         cudaStream_t stream = 0);

//! Traverse the LBVH using ropes.
template<class OutputOpT, class QueryOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         const QueryOpT& query,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int block_size,
                         cudaStream_t stream = 0);

/*
 * Templated function definitions should only be available in NVCC.
 */
#ifdef NVCC
namespace kernel
{
//! Kernel to compress LBVH for rope traversal
/*!
 * \param ctree Compressed LBVH.
 * \param transform Transformation operation.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 *
 * \tparam TransformOpT Type of operation for transforming cached primitive index.
 *
 * The bounding boxes and hierarchy of the LBVH are compressed into
 * (1) int4 / node. Each node holds the compressed bounds (2 ints),
 * the left child of the node (or primitive), and the rope to advance ahead.
 * The ropes are generated in this kernel by backtracking. The compression
 * converts the float bounds of the box into a 10-bit integer for each
 * component. The output \a bins size for the compression is done in a
 * conservative way so that on decompression, the bounds of the nodes are
 * never underestimated.
 *
 * The stored primitive may be transformed to a new value for more efficient caching for traversal.
 * The transformation is implemented by \a transform.
 */
template<class TransformOpT>
__global__ void lbvh_compress_ropes(LBVHCompressedData ctree,
                                    const TransformOpT transform,
                                    const LBVHData tree,
                                    const unsigned int N_internal,
                                    const unsigned int N_nodes)
    {
    // one thread per node
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (int)N_nodes)
        return;

    // load the tree extent for meshing
    __shared__ float3 tree_lo, tree_hi, tree_bininv;
    if (threadIdx.x == 0)
        {
        tree_lo = tree.lo[tree.root];
        tree_hi = tree.hi[tree.root];
        // compute box size, rounding up to ensure fully covered
        float3 L = make_float3(__fsub_ru(tree_hi.x, tree_lo.x),
                               __fsub_ru(tree_hi.y, tree_lo.y),
                               __fsub_ru(tree_hi.z, tree_lo.z));
        if (L.x <= 0.f) L.x = 1.0f;
        if (L.y <= 0.f) L.y = 1.0f;
        if (L.z <= 0.f) L.z = 1.0f;

        // round down the bin scale factor so that it always *underestimates* the offset
        tree_bininv = make_float3(__fdiv_rd(1023.f,L.x),
                                  __fdiv_rd(1023.f,L.y),
                                  __fdiv_rd(1023.f,L.z));
        }
    __syncthreads();

    // backtrack tree to find the first right ancestor of this node
    int rope = LBVHSentinel;
    int current = idx;
    while (current != tree.root && rope == LBVHSentinel)
        {
        int parent = tree.parent[current];
        int left = tree.left[parent];
        if (left == current)
            {
            // if this is a left node, then rope is determined to the right
            rope = tree.right[parent];
            }
        else
            {
            // otherwise, keep ascending the tree
            current = parent;
            }
        }

    // compress node data into one byte per box dim
    // low bounds are encoded relative to the low of the box, always rounding down
    const float3 lo = tree.lo[idx];
    const uint3 lo_bin = make_uint3((unsigned int)floorf(__fmul_rd(__fsub_rd(lo.x,tree_lo.x),tree_bininv.x)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(lo.y,tree_lo.y),tree_bininv.y)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(lo.z,tree_lo.z),tree_bininv.z)));
    const unsigned int lo_bin3 = (lo_bin.x << 20) +  (lo_bin.y << 10) + lo_bin.z;

    // high bounds are encoded relative to the high of the box, always rounding down
    const float3 hi = tree.hi[idx];
    const uint3 hi_bin = make_uint3((unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.x,hi.x),tree_bininv.x)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.y,hi.y),tree_bininv.y)),
                                    (unsigned int)floorf(__fmul_rd(__fsub_rd(tree_hi.z,hi.z),tree_bininv.z)));
    const unsigned int hi_bin3 = (hi_bin.x << 20) + (hi_bin.y << 10) + hi_bin.z;

    // node holds left child for internal nodes (>= 0) or primitive for leaf (< 0)
    int left_flag = (idx < N_internal) ? tree.left[idx] : ~transform(tree.primitive[idx-N_internal]);

    // stash all the data into one int4
    ctree.data[idx] = make_int4(lo_bin3, hi_bin3, left_flag, rope);

    // first thread writes out the compression values, rounding down bin size to ensure box bounds always expand even with floats
    if (idx == 0)
        {
        *ctree.lo = tree_lo;
        *ctree.hi = tree_hi;
        *ctree.bins = make_float3(__frcp_rd(tree_bininv.x),__frcp_rd(tree_bininv.y),__frcp_rd(tree_bininv.z));
        }
    }

//! Kernel to traverse the LBVH using ropes.
/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param query Query operation.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
 *
 * The LBVH is traversed using the rope scheme. In this method, the
 * test sphere always descends to the left child of an intersected node,
 * and advances to the next viable branch of the tree (along the rope) when
 * no overlap occurs. This is a stackless traversal scheme.
 *
 * The query volume for the traversal can be constructed using the \a query operation.
 * This operation is responsible for constructing the query volume, translating it,
 * and performing overlap operations with the BoundingBox volumes in the LBVH.
 *
 * Each query volume can optionally be translated by a set of \a d_images. The self-image
 * is automatically traversed and should not be included in \a d_images. Before
 * entering the traversal loop, each volume is translated and intersected against
 * the tree root. A set of bitflags is encoded for which images possibly overlap the
 * tree. (Some may only intersect in the self-image, while others may intersect multiple
 * times.) This is done first to avoid divergence within the traversal loop.
 * During traversal, an image processes the entire tree, and then advances to the next
 * image once traversal terminates. A maximum of 32 images is supported.
 */
template<class OutputOpT, class QueryOpT>
__global__ void lbvh_traverse_ropes(OutputOpT out,
                                    const LBVHCompressedData lbvh,
                                    const QueryOpT query,
                                    const Scalar3 *d_images,
                                    const unsigned int Nimages)
    {
    // one thread per test
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= query.size())
        return;
    typename OutputOpT::ThreadData result = out.setup(idx);

    // load tree compression sizes into shared memory
    __shared__ BoundingBox tree_box;
    __shared__ float3 tree_bins;
    if (threadIdx.x == 0)
        {
        tree_box = BoundingBox(*lbvh.lo, *lbvh.hi);
        tree_bins = *lbvh.bins;
        }
    __syncthreads();

    // query thread data
    const typename QueryOpT::ThreadData qdata = query.setup(idx);

    // find image flags against root before divergence
    unsigned int flags = 0;
    const int nbits = ((int)Nimages <= 32) ? Nimages : 32;
    for (unsigned int i=0; i < nbits; ++i)
        {
        const Scalar3 image = d_images[i];
        const typename QueryOpT::Volume q = query.get(qdata,image);
        if (query.overlap(q,tree_box)) flags |= 1u << i;
        }

    // stackless search
    typename QueryOpT::Volume q = query.get(qdata, make_scalar3(0,0,0));
    int node = lbvh.root;
    do
        {
        while (node != LBVHSentinel)
            {
            // load node and decompress bounds so that they always *expand*
            const int4 aabb = __ldg(lbvh.data + node);
            const unsigned int lo = aabb.x;
            const float3 lof = make_float3(__fadd_rd(tree_box.lo.x, __fmul_rd((lo >> 20) & 0x3ffu,tree_bins.x)),
                                           __fadd_rd(tree_box.lo.y, __fmul_rd((lo >> 10) & 0x3ffu,tree_bins.y)),
                                           __fadd_rd(tree_box.lo.z, __fmul_rd((lo      ) & 0x3ffu,tree_bins.z)));

            const unsigned int hi = aabb.y;
            const float3 hif = make_float3(__fsub_ru(tree_box.hi.x, __fmul_rd((hi >> 20) & 0x3ffu,tree_bins.x)),
                                           __fsub_ru(tree_box.hi.y, __fmul_rd((hi >> 10) & 0x3ffu,tree_bins.y)),
                                           __fsub_ru(tree_box.hi.z, __fmul_rd((hi      ) & 0x3ffu,tree_bins.z)));
            const int left = aabb.z;

            // advance to rope as a preliminary
            node = aabb.w;

            // if overlap, do work with primitive. otherwise, rope ahead
            if (query.overlap(q, BoundingBox(lof,hif)))
                {
                if(left < 0)
                    {
                    const int primitive = ~left;
                    if (query.refine(qdata,primitive))
                        out.process(result,primitive);
                    // leaf nodes always move to their rope
                    }
                else
                    {
                    // internal node takes left child
                    node = left;
                    }
                }
            } // end stackless search

        // look for the next image
        int image_bit = __ffs(flags);
        if (image_bit)
            {
            // shift the lsb by 1 to get the image index
            --image_bit;

            // move the sphere to the next image
            const Scalar3 image = d_images[image_bit];
            q = query.get(qdata, image);
            node = lbvh.root;

            // unset the bit from this image
            flags &= ~(1u << image_bit);
            }
        else
            {
            // no more images, quit
            break;
            }
        } while(true);

    out.finalize(result);
    }
} // end namespace kernel

/*!
 * \param ctree Compressed LBVH.
 * \param transform Transformation operation.
 * \param tree LBVH to compress.
 * \param N_internal Number of internal nodes in LBVH.
 * \param N_nodes Number of nodes in LBVH.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam TransformOpT Type of operation for transforming cached primitive index.
 *
 * \sa kernel::lbvh_compress_ropes
 */
//! Compress LBVH for rope traversal.
template<class TransformOpT>
void lbvh_compress_ropes(LBVHCompressedData ctree,
                         const TransformOpT& transform,
                         const LBVHData tree,
                         unsigned int N_internal,
                         unsigned int N_nodes,
                         unsigned int block_size,
                         cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_compress_ropes<TransformOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (N_nodes + run_block_size - 1)/run_block_size;
    kernel::lbvh_compress_ropes<<<num_blocks, run_block_size, 0, stream>>>
        (ctree, transform, tree, N_internal, N_nodes);
    }

/*!
 * \param out Output operation for intersected primitives.
 * \param lbvh Compressed LBVH data to traverse.
 * \param d_spheres Test spheres to intersect with LBVH.
 * \param d_images Image vectors to traverse for \a d_spheres.
 * \param Nimages Number of image vectors.
 * \param N Number of test spheres.
 * \param block_size Number of CUDA threads per block.
 * \param stream CUDA stream for kernel execution.
 *
 * \tparam OutputOpT The type of output operation.
 * \tparam QueryOpT The type of query operation.
 *
 * \sa kernel::lbvh_traverse_ropes
 */
template<class OutputOpT, class QueryOpT>
void lbvh_traverse_ropes(OutputOpT& out,
                         const LBVHCompressedData& lbvh,
                         const QueryOpT& query,
                         const Scalar3 *d_images,
                         unsigned int Nimages,
                         unsigned int block_size,
                         cudaStream_t stream)
    {
    // clamp block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)kernel::lbvh_traverse_ropes<OutputOpT,QueryOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = (block_size < max_block_size) ? block_size : max_block_size;

    const unsigned int num_blocks = (query.size() + run_block_size - 1)/run_block_size;
    kernel::lbvh_traverse_ropes<<<num_blocks, run_block_size, 0, stream>>>
        (out, lbvh, query, d_images, Nimages);
    }
#endif

} // end namespace gpu
} // end namespace neighbor

#endif // NEIGHBOR_LBVH_TRAVERSER_CUH_
