// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_LBVH_H_
#define NEIGHBOR_LBVH_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/Autotuner.h"
#include "hoomd/CachedAllocator.h"

#include "LBVH.cuh"

namespace neighbor
{

//! Linear bounding volume hierarchy
/*!
 * A linear bounding hierarchy (LBVH) is a binary tree structure that can be used for overlap
 * or collision detection. Briefly, a leaf node in the tree encloses a single primitive
 * object (currently assumed to be a point). The next layer of (internal) nodes in the tree
 * encloses two leaf nodes, while higher layers enclose two internal nodes. The
 * tree terminates in a single root node. The nodes are fit by axis-aligned bounding boxes.
 * The bounding boxes have lower and upper bounds in Cartesian coordinates that enclose
 * all primitives that are children of a node. The LBVH can be traversed by testing for
 * intersections of some test object against a node and then descending to the children of
 * intersected nodes until the primitives are reached.
 *
 * The LBVH class constructs such a hierarchy on the GPU using the ::build method. The
 * point primitives must be supplied as Scalar4s (defined in HOOMD-blue precision model).
 * Regardless of the precision of the primitives, the bounding boxes are stored in
 * single-precision in a way that preserves correctness of the tree. The build algorithm
 * is due to Karras with 30-bit Morton codes to sort primitives.
 *
 * The data needed for tree traversal can be accessed by the appropriate methods. It is
 * recommended to use the sorted primitive order for traversal for best performance
 * (see ::getPrimitives). Since different traversal schemes can be prescribed or additional
 * tree processing may occur, the traversal is delegated to a LBVHTraverser object.
 * The memory layout of the data arrays is such that all internal nodes precede all
 * leaf nodes, and the root node is node 0.
 *
 * For processing the LBVH in GPU kernels, it may be useful to obtain an object containing
 * only the raw pointers to the tree data (see LBVHData in LBVH.cuh). The caller must
 * construct such an object due to the multitude of different access modes that are possible
 * for the GPU data.
 */
class PYBIND11_EXPORT LBVH
    {
    public:
        //! Setup an unallocated LBVH
        LBVH(std::shared_ptr<const ExecutionConfiguration> exec_conf,
             cudaStream_t stream = 0);

        //! Destroy an LBVH
        ~LBVH();

        //! Build the LBVH
        template<class InsertOpT>
        void build(const InsertOpT& insert, const Scalar3 lo, const Scalar3 hi);

        //! Get the LBVH root node
        int getRoot() const
            {
            return m_root;
            }

        //! Get the number of primitives
        unsigned int getN() const
            {
            return m_N;
            }

        //! Get the number of internal nodes
        unsigned int getNInternal() const
            {
            return m_N_internal;
            }

        //! Get the total number of nodes
        unsigned int getNNodes() const
            {
            return m_N_nodes;
            }

        //! Get the array of parents of a given node
        const GlobalArray<int>& getParents() const
            {
            return m_parent;
            }

        //! Get the array of left children of a given node
        const GlobalArray<int>& getLeftChildren() const
            {
            return m_left;
            }

        //! Get the array of right children of a given node
        const GlobalArray<int>& getRightChildren() const
            {
            return m_right;
            }

        //! Get the lower bounds of the boxes enclosing a node
        const GlobalArray<float3>& getLowerBounds() const
            {
            return m_lo;
            }

        //! Get the upper bounds of the boxes enclosing a node
        const GlobalArray<float3>& getUpperBounds() const
            {
            return m_hi;
            }

        //! Get the original indexes of the primitives in each leaf node
        const GlobalArray<unsigned int>& getPrimitives() const
            {
            return m_sorted_indexes;
            }

        //! Set the kernel autotuner parameters
        /*!
         * \param enable If true, run the autotuners. If false, disable them.
         * \param period Number of builds between running the autotuners.
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            m_tune_gen_codes->setEnabled(enable);
            m_tune_gen_codes->setPeriod(period);

            m_tune_gen_tree->setEnabled(enable);
            m_tune_gen_tree->setPeriod(period);

            m_tune_bubble->setEnabled(enable);
            m_tune_bubble->setPeriod(period);
            }

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< HOOMD execution configuration
        cudaStream_t m_stream;  //! CUDA stream to build in

        int m_root;                 //!< Root index
        unsigned int m_N;           //!< Number of primitives in the tree
        unsigned int m_N_internal;  //!< Number of internal nodes in tree
        unsigned int m_N_nodes;     //!< Number of nodes in the tree

        GlobalArray<int> m_parent; //!< Parent node
        GlobalArray<int> m_left;   //!< Left child
        GlobalArray<int> m_right;  //!< Right child
        GlobalArray<float3> m_lo;  //!< Lower bound of AABB
        GlobalArray<float3> m_hi;  //!< Upper bound of AABB

        GlobalArray<unsigned int> m_codes;             //!< Morton codes
        GlobalArray<unsigned int> m_indexes;           //!< Primitive indexes
        GlobalArray<unsigned int> m_sorted_codes;      //!< Sorted morton codes
        GlobalArray<unsigned int> m_sorted_indexes;    //!< Sorted primitive indexes

        GlobalArray<unsigned int> m_locks; //!< Node locks for generating aabb hierarchy

        std::unique_ptr<Autotuner> m_tune_gen_codes;    //!< Autotuner for generating Morton codes kernel
        std::unique_ptr<Autotuner> m_tune_gen_tree;     //!< Autotuner for generating tree hierarchy kernel
        std::unique_ptr<Autotuner> m_tune_bubble;       //!< Autotuner for AABB bubble kernel

        //! Allocate
        void allocate(unsigned int N);
    };

/*!
 * \param insert The insert operation determining AABB extents of primitives
 * \param N Number of primitives
 * \param lo Lower bound of the scene
 * \param hi Upper bound of the scene
 *
 * \tparam InsertOpT the kind of insert operation
 *
 * The LBVH is constructed using the algorithm due to Karras using 30-bit Morton codes.
 * The caller should ensure that all \a points lie within \a lo and \a hi for best performance.
 * Points lying outside this range are clamped to it during the Morton code calculation, which
 * may lead to a low quality LBVH.
 *
 * \note
 * Currently, small LBVHs (`N` <= 2) are not implemented, and an error will be raised.
 */
template<class InsertOpT>
void LBVH::build(const InsertOpT& insert, const Scalar3 lo, const Scalar3 hi)
    {
    const unsigned int N = insert.size();

    // resize memory for the tree
    allocate(N);

    // if N = 0, don't do anything and quit, since this is an empty lbvh
    if (N == 0) return;

    // single-particle just needs a small amount of data
    if (N == 1)
        {
        ArrayHandle<int> d_parent(m_parent, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);
        ArrayHandle<float3> d_lo(m_lo, access_location::device, access_mode::overwrite);
        ArrayHandle<float3> d_hi(m_hi, access_location::device, access_mode::overwrite);

        neighbor::gpu::LBVHData tree;
        tree.parent = d_parent.data;
        tree.left = NULL;
        tree.right = NULL;
        tree.primitive = d_sorted_indexes.data;
        tree.lo = d_lo.data;
        tree.hi = d_hi.data;
        tree.root = m_root;

        neighbor::gpu::lbvh_one_primitive(tree, insert, m_stream);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        return;
        }

    // calculate morton codes
        {
        ArrayHandle<unsigned int> d_codes(m_codes, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::overwrite);

        m_tune_gen_codes->begin();
        neighbor::gpu::lbvh_gen_codes(d_codes.data, d_indexes.data, insert, lo, hi, m_N, m_tune_gen_codes->getParam(), m_stream);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_gen_codes->end();
        }

    // sort morton codes
        {
        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_codes(m_codes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_codes(m_sorted_codes, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_indexes(m_indexes, access_location::device, access_mode::readwrite);
            ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::overwrite);

            void *d_tmp = NULL;
            size_t tmp_bytes = 0;
            neighbor::gpu::lbvh_sort_codes(d_tmp,
                                           tmp_bytes,
                                           d_codes.data,
                                           d_sorted_codes.data,
                                           d_indexes.data,
                                           d_sorted_indexes.data,
                                           m_N,
                                           m_stream);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void *)d_alloc();

            swap = neighbor::gpu::lbvh_sort_codes(d_tmp,
                                                  tmp_bytes,
                                                  d_codes.data,
                                                  d_sorted_codes.data,
                                                  d_indexes.data,
                                                  d_sorted_indexes.data,
                                                  m_N,
                                                  m_stream);
            }
        if (swap.x) m_sorted_codes.swap(m_codes);
        if (swap.y) m_sorted_indexes.swap(m_indexes);
        }

    // process hierarchy and bubble aabbs
        {
        ArrayHandle<int> d_parent(m_parent, access_location::device, access_mode::overwrite);
        ArrayHandle<int> d_left(m_left, access_location::device, access_mode::overwrite);
        ArrayHandle<int> d_right(m_right, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_sorted_indexes(m_sorted_indexes, access_location::device, access_mode::read);
        ArrayHandle<float3> d_lo(m_lo, access_location::device, access_mode::overwrite);
        ArrayHandle<float3> d_hi(m_hi, access_location::device, access_mode::overwrite);

        neighbor::gpu::LBVHData tree;
        tree.parent = d_parent.data;
        tree.left = d_left.data;
        tree.right = d_right.data;
        tree.primitive = d_sorted_indexes.data;
        tree.lo = d_lo.data;
        tree.hi = d_hi.data;
        tree.root = m_root;

        // generate the tree hierarchy
        ArrayHandle<unsigned int> d_sorted_codes(m_sorted_codes, access_location::device, access_mode::read);

        m_tune_gen_tree->begin();
        neighbor::gpu::lbvh_gen_tree(tree, d_sorted_codes.data, m_N, m_tune_gen_tree->getParam(), m_stream);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_gen_tree->end();

        // bubble up the aabbs
        ArrayHandle<unsigned int> d_locks(m_locks, access_location::device, access_mode::overwrite);

        m_tune_bubble->begin();
        neighbor::gpu::lbvh_bubble_aabbs(tree, insert, d_locks.data, m_N, m_tune_bubble->getParam(), m_stream);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tune_bubble->end();
        }
    }

} // end namespace neighbor

#endif // NEIGHBOR_LBVH_H_
