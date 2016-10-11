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
            m_level = ManagedArray<unsigned int>(m_num_nodes,managed);
            m_isleft = ManagedArray<unsigned int>(m_num_nodes,managed);
            m_parent = ManagedArray<unsigned int>(m_num_nodes,managed);
            m_rcl = ManagedArray<unsigned int>(m_num_nodes,managed);
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

            // update auxillary information for tandem traversal
            updateRCL(0, tree, 0, true, m_num_nodes, 0);
            }
        #endif

        //! Returns number of nodes in tree
        DEVICE unsigned int getNumNodes() const
            {
            return m_num_nodes;
            }

        #if 0
        //! Fetch the next node in the tree and test against overlap
        /*! The method maintains it internal state in a user-supplied variable cur_node
         *
         * \param obb Query bounding box
         * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous call
         * \param particles List of particles returned (array of at least capacity length), -1 means no particle
         * \returns true if the current node overlaps and is a leaf node
         */
        DEVICE inline bool queryNode(const OBB& obb, unsigned int &cur_node, int *particles) const
            {
            OBB node_obb(m_lower[cur_node],m_upper[cur_node]);

            bool leaf = false;
            if (overlap(node_obb, obb))
                {
                // is this node a leaf node?
                if (m_left[cur_node] == INVALID_NODE)
                    {
                    for (unsigned int i = 0; i < capacity; i++)
                        particles[i] = m_particles[cur_node*capacity+i];
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
        #endif

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


        DEVICE inline unsigned int getLevel(unsigned int node) const
            {
            return m_level[node];
            }

        DEVICE inline unsigned int getLeftChild(unsigned int node) const
            {
            return m_left[node];
            }

        DEVICE inline bool isLeftChild(unsigned int node) const
            {
            return m_isleft[node];
            }

        DEVICE inline unsigned int getParent(unsigned int node) const
            {
            return m_parent[node];
            }

        DEVICE inline unsigned int getRCL(unsigned int node) const
            {
            return m_rcl[node];
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

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param load If true, copy data to pointer, otherwise increment only
         */
        HOSTDEVICE void load_shared(char *& ptr, bool load=true) const
            {
//            m_center.load_shared(ptr, load);
//            m_lengths.load_shared(ptr, load);
//            m_rotation.load_shared(ptr, load);

//            m_level.load_shared(ptr, load);
//            m_isleft.load_shared(ptr, load);
//            m_parent.load_shared(ptr, load);
//            m_rcl.load_shared(ptr, load);
            m_left.load_shared(ptr, load);
            m_skip.load_shared(ptr, load);

//            m_leaf_ptr.load_shared(ptr, load);
//            m_leaf_obb_ptr.load_shared(ptr, load);
//            m_particles.load_shared(ptr, load);
            }

    protected:
        #ifndef NVCC
        void updateRCL(unsigned int idx, const obb_tree_type& tree, unsigned int level, bool left,
             unsigned int parent_idx, unsigned int rcl)
            {
            if (!isLeaf(idx))
                {
                unsigned int left_idx = tree.getNodeLeft(idx);;
                unsigned int right_idx = tree.getNode(idx).right;

                updateRCL(left_idx, tree, level+1, true, idx, 0);
                updateRCL(right_idx, tree, level+1, false, idx, rcl+1);
                }
            m_level[idx] = level;
            m_isleft[idx] = left;
            m_parent[idx] = parent_idx;
            m_rcl[idx] = rcl;
            }
        #endif

    private:
        ManagedArray<vec3<OverlapReal> > m_center;
        ManagedArray<vec3<OverlapReal> > m_lengths;
        ManagedArray<rotmat3<OverlapReal> > m_rotation;

        ManagedArray<unsigned int> m_level;   //!< Depth
        ManagedArray<unsigned int> m_isleft;  //!< True if this node is a left node
        ManagedArray<unsigned int> m_parent;  //!< Pointer to parent
        ManagedArray<unsigned int> m_rcl;     //!< Right child level

        ManagedArray<unsigned int> m_leaf_ptr; //!< Pointer to leaf node contents
        ManagedArray<unsigned int> m_leaf_obb_ptr; //!< Pointer to leaf node OBBs
        ManagedArray<unsigned int> m_particles;        //!< Stores the leaf nodes' indices

        ManagedArray<unsigned int> m_left;    //!< Left nodes
        ManagedArray<unsigned int> m_skip;    //!< Skip intervals

        unsigned int m_num_nodes;             //!< Number of nodes in the tree
        unsigned int m_num_leaves;            //!< Number of leaf nodes
    };

//! Test a subtree against a leaf node during a tandem traversal
template<class Shape, class Tree>
DEVICE inline bool test_subtree(const vec3<OverlapReal>& r_ab,
                                const Shape& s0,
                                const Shape& s1,
                                const Tree& tree_a,
                                const Tree& tree_b,
                                unsigned int leaf_node,
                                unsigned int cur_node,
                                unsigned int end_idx)
    {
    // get the obb of the leaf node
    hpmc::detail::OBB obb = tree_a.getOBB(leaf_node);
    obb.affineTransform(conj(quat<OverlapReal>(s1.orientation))*quat<OverlapReal>(s0.orientation),
        rotate(conj(quat<OverlapReal>(s1.orientation)),-r_ab));

    while (cur_node != end_idx)
        {
        hpmc::detail::OBB node_obb = tree_b.getOBB(cur_node);

        bool skip = false;

        if (detail::overlap(obb,node_obb))
            {
            if (tree_b.isLeaf(cur_node))
                {
                if (test_narrow_phase_overlap(r_ab, s0, s1, leaf_node, cur_node)) return true;
                }
            }
        else
            {
            skip = true;
            }
        tree_b.advanceNode(cur_node, skip);
        }
    return false;
    }

//! Move up during a tandem traversal, alternating between trees a and b
/*! Adapted from: "Stackless BVH Collision Detection for Physical Simulation" by
    Jesper Damkjaer, damkjaer@diku.edu, http://image.diku.dk/projects/media/jesper.damkjaer.07.pdf
 */
template<class Tree>
DEVICE inline void moveUp(const Tree& tree_a, unsigned int& cur_node_a, const Tree& tree_b, unsigned int& cur_node_b)
    {
    unsigned int level_a = tree_a.getLevel(cur_node_a);
    unsigned int level_b = tree_b.getLevel(cur_node_b);

    if (level_a == level_b)
        {
        bool a_is_left_child = tree_a.isLeftChild(cur_node_a);
        bool b_is_left_child = tree_b.isLeftChild(cur_node_b);
        if (a_is_left_child)
            {
            tree_a.advanceNode(cur_node_a, true);
            return;
            }
        if (!a_is_left_child && b_is_left_child)
            {
            cur_node_a = tree_a.getParent(cur_node_a);
            tree_b.advanceNode(cur_node_b, true);
            return;
            }
        if (!a_is_left_child && !b_is_left_child)
            {
            unsigned int rcl_a = tree_a.getRCL(cur_node_a);
            unsigned int rcl_b = tree_b.getRCL(cur_node_b);
            if (rcl_a <= rcl_b)
                {
                tree_a.advanceNode(cur_node_a, true);
                // LevelUp
                while (rcl_a)
                    {
                    cur_node_b = tree_b.getParent(cur_node_b);
                    rcl_a--;
                    }
                }
            else
                {
                // LevelUp
                rcl_b++;
                while (rcl_b)
                    {
                    cur_node_a = tree_a.getParent(cur_node_a);
                    rcl_b--;
                    }
                tree_b.advanceNode(cur_node_b, true);
                }
            return;
            }
        } // end if level_a == level_b
    else
        {
        bool a_is_left_child = tree_a.isLeftChild(cur_node_a);
        bool b_is_left_child = tree_b.isLeftChild(cur_node_b);

        if (b_is_left_child)
            {
            tree_b.advanceNode(cur_node_b, true);
            return;
            }
        if (a_is_left_child)
            {
            tree_a.advanceNode(cur_node_a, true);
            cur_node_b = tree_b.getParent(cur_node_b);
            return;
            }
        if (!a_is_left_child && !b_is_left_child)
            {
            unsigned int rcl_a = tree_a.getRCL(cur_node_a);
            unsigned int rcl_b = tree_b.getRCL(cur_node_b);

            if (rcl_a <= rcl_b-1)
                {
                tree_a.advanceNode(cur_node_a, true);
                // LevelUp
                rcl_a++;
                while (rcl_a)
                    {
                    cur_node_b = tree_b.getParent(cur_node_b);
                    rcl_a--;
                    }
                }
            else
                {
                // LevelUp
                while (rcl_b)
                    {
                    cur_node_a = tree_a.getParent(cur_node_a);
                    rcl_b--;
                    }
                tree_b.advanceNode(cur_node_b, true);
                }
            return;
            }
        }
    }

}; // end namespace detail

}; // end namespace hpmc

#endif // __GPU_TREE_H__
