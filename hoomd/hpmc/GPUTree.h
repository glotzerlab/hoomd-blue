// Copyright (c) 2009-2019 The Regents of the University of Michigan
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
class GPUTree
    {
    public:
        //! Empty constructor
        GPUTree()
            : m_num_nodes(0), m_num_leaves(0), m_leaf_capacity(0)
            { }

        #ifndef NVCC
        //! Constructor
        /*! \param tree OBBTree to construct from
         *  \param managed True if we use CUDA managed memory
         */
        GPUTree(const OBBTree &tree, bool managed=false)
            {
            // allocate
            m_num_nodes = tree.getNumNodes();

            m_center = ManagedArray<vec3<OverlapReal> >(m_num_nodes, managed);
            m_lengths = ManagedArray<vec3<OverlapReal> >(m_num_nodes,managed);
            m_rotation = ManagedArray<quat<OverlapReal> >(m_num_nodes,managed);
            m_mask = ManagedArray<unsigned int>(m_num_nodes,managed);
            m_is_sphere = ManagedArray<unsigned int>(m_num_nodes,managed);
            m_left = ManagedArray<unsigned int>(m_num_nodes, managed);
            m_escape = ManagedArray<unsigned int>(m_num_nodes, managed);
            m_ancestors = ManagedArray<unsigned int>(m_num_nodes, managed);
            m_leaf_ptr = ManagedArray<unsigned int>(m_num_nodes+1, managed);

            unsigned int n = 0;
            m_num_leaves = 0;

            // load data from AABTree
            for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
                {
                m_left[i] = tree.getNodeLeft(i);
                m_escape[i] = tree.getEscapeIndex(i);

                m_center[i] = tree.getNodeOBB(i).getPosition();
                m_rotation[i] = tree.getNodeOBB(i).rotation;
                m_lengths[i] = tree.getNodeOBB(i).lengths;
                m_mask[i] = tree.getNodeOBB(i).mask;
                m_is_sphere[i] = tree.getNodeOBB(i).isSphere();

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

            m_leaf_capacity = tree.getLeafNodeCapacity();

            // recursively initialize ancestor indices
            initializeAncestorCounts(0, tree, 0);
            }
        #endif

        //! Returns number of nodes in tree
        DEVICE unsigned int getNumNodes() const
            {
            return m_num_nodes;
            }

        #ifndef NVCC
        //! Initialize the ancestor count index
        void initializeAncestorCounts(unsigned int idx, const OBBTree& tree, unsigned int ancestors)
            {
            if (!isLeaf(idx))
                {
                unsigned int left_idx = tree.getNodeLeft(idx);;
                unsigned int right_idx = tree.getNode(idx).right;

                initializeAncestorCounts(left_idx, tree, 0);
                initializeAncestorCounts(right_idx, tree, ancestors+1);
                }

            m_ancestors[idx] = ancestors;
            }
        #endif

        //! Fetch the next node in the tree and test against overlap
        /*! The method maintains it internal state in a user-supplied variable cur_node
         *
         * \param obb Query bounding box
         * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous call
         * \returns true if the current node overlaps and is a leaf node
         */
        DEVICE inline bool queryNode(const OBB& obb, unsigned int &cur_node) const
            {
            OBB node_obb(getOBB(cur_node));

            bool leaf = false;
            if (overlap(node_obb, obb))
                {
                unsigned int left_child = getLeftChild(cur_node);

                // is this node a leaf node?
                if (left_child == OBB_INVALID_NODE)
                    {
                    leaf = true;
                    }
                else
                    {
                    cur_node = left_child;
                    return false;
                    }
                }

            // escape
            cur_node = m_escape[cur_node];

            return leaf;
            }

        //! Fetch the next node in the tree and test against overlap with a ray
        /*! The method maintains it internal state in a user-supplied variable cur_node
         * The ray equation is R(t) = p + t*d (t>=0)
         *
         * \param p origin of ray
         * \param d direction of ray
         * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous call
         * \param abs_tol an absolute tolerance
         * \returns true if the current node overlaps and is a leaf node
         */
        DEVICE inline bool queryRay(const vec3<OverlapReal>& p, const vec3<OverlapReal>& d, unsigned int &cur_node, OverlapReal abs_tol) const
            {
            OBB node_obb(getOBB(cur_node));

            OverlapReal t;
            vec3<OverlapReal> q;
            bool leaf = false;
            if (IntersectRayOBB(p,d,node_obb,t,q, abs_tol))
                {
                // is this node a leaf node?
                unsigned int left_child = getLeftChild(cur_node);
                if (left_child == OBB_INVALID_NODE)
                    {
                    leaf = true;
                    }
                else
                    {
                    cur_node = left_child;
                    return false;
                    }
                }

            // escape
            cur_node = m_escape[cur_node];

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

        DEVICE inline unsigned int getEscapeIndex(unsigned int node) const
            {
            return m_escape[node];
            }

        DEVICE inline unsigned int getNumAncestors(unsigned int node) const
            {
            return m_ancestors[node];
            }

        DEVICE inline OBB getOBB(unsigned int idx) const
            {
            OBB obb;
            obb.center = m_center[idx];
            obb.lengths = m_lengths[idx];
            obb.rotation = m_rotation[idx];
            obb.mask = m_mask[idx];
            obb.is_sphere = m_is_sphere[idx];
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
            m_mask.attach_to_stream(stream);
            m_is_sphere.attach_to_stream(stream);

            m_left.attach_to_stream(stream);
            m_escape.attach_to_stream(stream);
            m_ancestors.attach_to_stream(stream);

            m_leaf_ptr.attach_to_stream(stream);
            m_leaf_obb_ptr.attach_to_stream(stream);
            m_particles.attach_to_stream(stream);
            }
        #endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
            {
            m_center.load_shared(ptr, available_bytes);
            m_lengths.load_shared(ptr, available_bytes);
            m_rotation.load_shared(ptr, available_bytes);
            m_mask.load_shared(ptr, available_bytes);
            m_is_sphere.load_shared(ptr, available_bytes);

            m_left.load_shared(ptr, available_bytes);
            m_escape.load_shared(ptr, available_bytes);
            m_ancestors.load_shared(ptr, available_bytes);

            m_leaf_ptr.load_shared(ptr, available_bytes);
            m_leaf_obb_ptr.load_shared(ptr, available_bytes);
            m_particles.load_shared(ptr, available_bytes);
            }

        //! Get the capacity of leaf nodes
        unsigned int getLeafNodeCapacity() const
            {
            return m_leaf_capacity;
            }

    private:
        ManagedArray<vec3<OverlapReal> > m_center;
        ManagedArray<vec3<OverlapReal> > m_lengths;
        ManagedArray<quat<OverlapReal> > m_rotation;
        ManagedArray<unsigned int> m_mask;
        ManagedArray<unsigned int> m_is_sphere;

        ManagedArray<unsigned int> m_leaf_ptr; //!< Pointer to leaf node contents
        ManagedArray<unsigned int> m_leaf_obb_ptr; //!< Pointer to leaf node OBBs
        ManagedArray<unsigned int> m_particles;        //!< Stores the leaf nodes' indices

        ManagedArray<unsigned int> m_left;    //!< Left nodes
        ManagedArray<unsigned int> m_escape;  //!< Escape indices
        ManagedArray<unsigned int> m_ancestors;  //!< Number of right-most ancestors

        unsigned int m_num_nodes;             //!< Number of nodes in the tree
        unsigned int m_num_leaves;            //!< Number of leaf nodes
        unsigned int m_leaf_capacity;         //!< Capacity of OBB leaf nodes
    };


// Tandem stack traversal routines
// from: A Binary Stack Tandem Traversal and an Ancestor Counter Data Structure for GPU friendly Bounding Volume
// Damkjær, Jesper and Erleben, Kenny
// Proceedings Workshop in Virtual Reality Interactions and Physical Simulation "VRIPHYS" (2009)
// http://dx.doi.org/10.2312/PE/vriphys/vriphys09/115-124

//! Compute how many ascents are necessary to reach a non right-most child
/*! \param a_count Ancestor count in tree a
    \param b_count Ancestor count in tree b
    \param Binary stack
    \param a_ascent Number of ascents in a (return variable)
    \param b_ascent Number of ascents in b (return variable)
 */
DEVICE inline void findAscent(unsigned int a_count, unsigned int b_count, unsigned long int &stack,
    unsigned int& a_ascent, unsigned int& b_ascent)
    {
    a_ascent = 0;
    b_ascent = 0;

    while (true)
        {
        if ((stack & 1) == 0) // top of stack == A?
            {
            if (a_count > 0)
                {
                stack >>= 1; // pop
                a_count--;
                a_ascent++;
                }
            else
                return;
            }
        else
            {
            if (b_count > 0)
                {
                stack >>= 1; // pop
                b_count--;
                b_ascent++;
                }
            else
                return;
            }
        } // end while
    }

//! Traverse a binary hierarchy
/*! Returns true if an intersecting pair of leaf OBB's has been found
 * \param a First tree
 * \param b Second tree
 * \param cur_node_a Current node in first tree
 * \param cur_node_b Current node in second tree
 * \param a binary stack realized as an integer
 * \param obb_a OBB from first tree corresponding to cur_node_a
 * \param obb_b OBB from second tree corresponding to cur_node_b
 *
 * This function supposed to be called from a while-loop:
 *
 * unsigned long int stack = 0;
 * // load OBBs for the two nodes
 * obb_a = ...
 * // transform OBB a into B's frame
 * ...
 * obb_b = ...
 *
 * while (cur_node_a != a.tree.getNumNodes() && cur_node_b != b.tree.getNumNodes())
 *     {
 *     query_node_a = cur_node_a;
 *     query_node_b = cur_node_b;
 *     if (traverseBinaryStack(a, b, cur_node_a, cur_node_b, stack, obb_a, obb_b, ..))
 *            test_narrow_phase(a, b, query_node_a, query_node_b, ...)
 *     }
 */
DEVICE inline bool traverseBinaryStack(const GPUTree& a, const GPUTree &b, unsigned int& cur_node_a, unsigned int& cur_node_b,
    unsigned long int &stack, OBB& obb_a, OBB& obb_b, const quat<OverlapReal>& q, const vec3<OverlapReal>& dr)
    {
    bool leaf = false;
    bool ascend = true;

    unsigned int old_a = cur_node_a;
    unsigned int old_b = cur_node_b;

    if (overlap(obb_a, obb_b))
        {
        if (a.isLeaf(cur_node_a) && b.isLeaf(cur_node_b))
            {
            leaf = true;
            }
        else
            {
            // descend into subtree with larger volume first (unless there are no children)
            bool descend_A = obb_a.getVolume() > obb_b.getVolume() ? !a.isLeaf(cur_node_a) : b.isLeaf(cur_node_b);

            if (descend_A)
                {
                cur_node_a = a.getLeftChild(cur_node_a);
                stack <<= 1; // push A
                }
            else
                {
                cur_node_b = b.getLeftChild(cur_node_b);
                stack <<= 1; stack |= 1; // push B
                }
            ascend = false;
            }
        }

    if (ascend)
        {
        // ascend in tree
        unsigned int a_count = a.getNumAncestors(cur_node_a);
        unsigned int b_count = b.getNumAncestors(cur_node_b);

        unsigned int a_ascent, b_ascent;
        findAscent(a_count, b_count, stack, a_ascent, b_ascent);

        if ((stack & 1) == 0) // top of stack == A
            {
            cur_node_a = a.getEscapeIndex(cur_node_a);

            // ascend in B, using post-order indexing
            cur_node_b -= b_ascent;
            }
        else
            {
            // ascend in A, using post-order indexing
            cur_node_a -= a_ascent;
            cur_node_b = b.getEscapeIndex(cur_node_b);
            }
        }
    if (cur_node_a < a.getNumNodes() && cur_node_b < b.getNumNodes())
        {
        // pre-fetch OBBs
        if (old_a != cur_node_a)
            {
            obb_a = a.getOBB(cur_node_a);
            obb_a.affineTransform(q, dr);
            }
        if (old_b != cur_node_b)
            obb_b = b.getOBB(cur_node_b);
        }

    return leaf;
    }

}; // end namespace detail

}; // end namespace hpmc

#endif // __GPU_TREE_H__
