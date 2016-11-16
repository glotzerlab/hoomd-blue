// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "OBBTree.h"

#ifndef __GPU_TREE_H__
#define __GPU_TREE_H__

// need to declare these class methods with appropriate qualifiers when building in nvcc
#ifdef NVCC
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#ifndef NVCC
#include <sstream>
#endif

#include "hoomd/ManagedArray.h"

namespace hpmc
{

namespace detail
{

//! Adapter class to AABTree for query on the GPU
template<unsigned int node_capacity>
class GPUTree
    {
    public:
        #ifndef NVCC
        typedef OBBTree<node_capacity> obb_tree_type;
        #endif

        enum { capacity = node_capacity } Enum;

        //! Empty constructor
        GPUTree()
            : m_num_nodes(0), m_num_leaves(0)
            { }

        #ifndef NVCC
        //! Constructor
        /*! \param tree OBBTree to construct from
         *  \param managed True if we use CUDA managed memory
         */
        GPUTree(const obb_tree_type &tree, bool managed=false)
            {
            // allocate
            m_num_nodes = tree.getNumNodes();

            m_center = ManagedArray<vec3<OverlapReal> >(m_num_nodes, managed);
            m_lengths = ManagedArray<vec3<OverlapReal> >(m_num_nodes,managed);
            m_rotation = ManagedArray<rotmat3<OverlapReal> >(m_num_nodes,managed);
            m_left = ManagedArray<unsigned int>(m_num_nodes, managed);
            m_skip = ManagedArray<unsigned int>(m_num_nodes, managed);
            m_leaf_ptr = ManagedArray<unsigned int>(m_num_nodes+1, managed);

            unsigned int n = 0;
            m_num_leaves = 0;

            // load data from AABTree
            for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
                {
                m_left[i] = tree.getNodeLeft(i);
                m_skip[i] = tree.getNodeSkip(i);

                m_center[i] = tree.getNodeOBB(i).getPosition();
                m_rotation[i] = tree.getNodeOBB(i).rotation;
                m_lengths[i] = tree.getNodeOBB(i).lengths;

                m_leaf_ptr[i] = n;
                n += tree.getNodeNumParticles(i);

                if (m_left[i] == OBB_INVALID_NODE)
                    {
                    m_num_leaves++;
                    }
                }
            m_leaf_ptr[tree.getNumNodes()] = n;

            m_leaf_obb_ptr = ManagedArray<unsigned int>(m_num_leaves, managed);
            m_num_leaves = 0;
            for (unsigned int i =0; i < tree.getNumNodes(); ++i)
                {
                if (m_left[i] == OBB_INVALID_NODE)
                    {
                    m_leaf_obb_ptr[m_num_leaves++] = i;
                    }
                }

            m_particles = ManagedArray<unsigned int>(n, managed);

            for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
                {
                for (unsigned int j = 0; j < tree.getNodeNumParticles(i); ++j)
                    {
                    m_particles[m_leaf_ptr[i]+j] = tree.getNodeParticle(i,j);
                    }
                }
            }
        #endif

        //! Returns number of nodes in tree
        DEVICE unsigned int getNumNodes() const
            {
            return m_num_nodes;
            }

        //! Fetch the next node in the tree and test against overlap
        /*! The method maintains it internal state in a user-supplied variable cur_node
         *
         * \param obb Query bounding box
         * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous call
         * \param particles List of particles returned (array of at least capacity length), -1 means no particle
         * \returns true if the current node overlaps and is a leaf node
         */
        DEVICE inline bool queryNode(const OBB& obb, unsigned int &cur_node) const
            {
            OBB node_obb(getOBB(cur_node));

            bool leaf = false;
            if (overlap(node_obb, obb))
                {
                // is this node a leaf node?
                if (m_left[cur_node] == OBB_INVALID_NODE)
                    {
                    leaf = true;
                    }
                }
            else
                {
                // skip ahead
                cur_node += m_skip[cur_node];
                }

            // advance cur_node
            cur_node ++;

            return leaf;
            }

        //! Test if a given index is a leaf node
        DEVICE inline bool isLeaf(unsigned int idx) const
            {
            return (m_left[idx] == OBB_INVALID_NODE);
            }

        //! Return the ith leaf node
        DEVICE inline unsigned int getLeafNode(unsigned int i) const
            {
            return m_leaf_obb_ptr[i];
            }

        //! Return the number of leaf nodes
        DEVICE inline unsigned int getNumLeaves() const
            {
            return m_num_leaves;
            }

        DEVICE inline unsigned int getParticle(unsigned int node, unsigned int i) const
            {
            return m_particles[m_leaf_ptr[node]+i];
            }

        DEVICE inline int getNumParticles(unsigned int node) const
            {
            return m_leaf_ptr[node+1] - m_leaf_ptr[node];
            }


        DEVICE inline unsigned int getLeftChild(unsigned int node) const
            {
            return m_left[node];
            }

        DEVICE inline void advanceNode(unsigned int &cur_node, bool skip) const
            {
            if (skip) cur_node += m_skip[cur_node];
            cur_node++;
            }

        DEVICE inline OBB getOBB(unsigned int idx) const
            {
            OBB obb;
            obb.center = m_center[idx];
            obb.lengths = m_lengths[idx];
            obb.rotation = m_rotation[idx];
            return obb;
            }

        #ifdef ENABLE_CUDA
        //! Attach managed memory to CUDA stream
        void attach_to_stream(cudaStream_t stream) const
            {
            // attach managed memory arrays to stream
            m_center.attach_to_stream(stream);
            m_lengths.attach_to_stream(stream);
            m_rotation.attach_to_stream(stream);

            m_left.attach_to_stream(stream);
            m_skip.attach_to_stream(stream);

            m_leaf_ptr.attach_to_stream(stream);
            m_leaf_obb_ptr.attach_to_stream(stream);
            m_particles.attach_to_stream(stream);
            }
        #endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param load If true, copy data to pointer, otherwise increment only
            \param ptr_max Maximum address in shared memory
         */
        HOSTDEVICE void load_shared(char *& ptr, bool load, char *ptr_max) const
            {
            m_center.load_shared(ptr, load, ptr_max);
            m_lengths.load_shared(ptr, load, ptr_max);
            m_rotation.load_shared(ptr, load, ptr_max);

            m_left.load_shared(ptr, load, ptr_max);
            m_skip.load_shared(ptr, load, ptr_max);

            m_leaf_ptr.load_shared(ptr, load, ptr_max);
            m_leaf_obb_ptr.load_shared(ptr, load, ptr_max);
            m_particles.load_shared(ptr, load, ptr_max);
            }

    private:
        ManagedArray<vec3<OverlapReal> > m_center;
        ManagedArray<vec3<OverlapReal> > m_lengths;
        ManagedArray<rotmat3<OverlapReal> > m_rotation;

        ManagedArray<unsigned int> m_leaf_ptr; //!< Pointer to leaf node contents
        ManagedArray<unsigned int> m_leaf_obb_ptr; //!< Pointer to leaf node OBBs
        ManagedArray<unsigned int> m_particles;        //!< Stores the leaf nodes' indices

        ManagedArray<unsigned int> m_left;    //!< Left nodes
        ManagedArray<unsigned int> m_skip;    //!< Skip intervals

        unsigned int m_num_nodes;             //!< Number of nodes in the tree
        unsigned int m_num_leaves;            //!< Number of leaf nodes
    };


}; // end namespace detail

}; // end namespace hpmc

#endif // __GPU_TREE_H__
