// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "OBB.h"

#ifndef __HIPCC__
#include "OBBTree.h"
#endif

#include "hoomd/ManagedArray.h"

#ifndef __GPU_TREE_H__
#define __GPU_TREE_H__

// need to declare these class methods with appropriate qualifiers when building in nvcc
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#ifndef __HIPCC__
#include <sstream>
#endif

#include "hoomd/ManagedArray.h"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Adapter class to AABTree for query on the GPU
class GPUTree
    {
    public:
    //! Empty constructor
    HOSTDEVICE GPUTree() : m_num_nodes(0), m_num_leaves(0), m_leaf_capacity(0) { }

#ifndef __HIPCC__
    //! Constructor
    /*! \param tree OBBTree to construct from
     *  \param managed True if we use CUDA managed memory
     */
    GPUTree(const OBBTree& tree, bool managed = false)
        {
        // allocate
        m_num_nodes = tree.getNumNodes();

        m_center = ManagedArray<vec3<ShortReal>>(m_num_nodes, managed);
        m_lengths = ManagedArray<vec3<ShortReal>>(m_num_nodes, managed);
        m_rotation = ManagedArray<quat<ShortReal>>(m_num_nodes, managed);
        m_mask = ManagedArray<unsigned int>(m_num_nodes, managed);
        m_is_sphere = ManagedArray<unsigned int>(m_num_nodes, managed);
        m_left = ManagedArray<unsigned int>(m_num_nodes, managed);
        m_escape = ManagedArray<unsigned int>(m_num_nodes, managed);
        m_ancestors = ManagedArray<unsigned int>(m_num_nodes, managed);
        m_leaf_ptr = ManagedArray<unsigned int>(m_num_nodes + 1, managed);

        unsigned int n = 0;
        m_num_leaves = 0;

        // load data from OBBTree
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
        for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
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
                m_particles[m_leaf_ptr[i] + j] = tree.getNodeParticle(i, j);
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

#ifndef __HIPCC__
    //! Initialize the ancestor count index
    void initializeAncestorCounts(unsigned int idx, const OBBTree& tree, unsigned int ancestors)
        {
        if (!isLeaf(idx))
            {
            unsigned int left_idx = tree.getNodeLeft(idx);
            ;
            unsigned int right_idx = tree.getNode(idx).right;

            initializeAncestorCounts(left_idx, tree, 0);
            initializeAncestorCounts(right_idx, tree, ancestors + 1);
            }

        m_ancestors[idx] = ancestors;
        }
#endif

    //! Fetch the next node in the tree and test against overlap
    /*! The method maintains it internal state in a user-supplied variable cur_node
     *
     * \param obb Query bounding box
     * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous
     * call \returns true if the current node overlaps and is a leaf node
     */
    DEVICE inline bool queryNode(const OBB& obb, unsigned int& cur_node) const
        {
        OBB node_obb(getOBB(cur_node));

        bool leaf = false;
        if (overlap(node_obb, obb))
            {
            unsigned int left_child = getLeftChild(cur_node);

            // is this node a leaf node?
            if (left_child == 0xffffffff)
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
     * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous
     * call \param abs_tol an absolute tolerance \returns true if the current node overlaps and is a
     * leaf node
     */
    DEVICE inline bool queryRay(const vec3<ShortReal>& p,
                                const vec3<ShortReal>& d,
                                unsigned int& cur_node,
                                ShortReal abs_tol) const
        {
        OBB node_obb(getOBB(cur_node));

        ShortReal t;
        vec3<ShortReal> q;
        bool leaf = false;
        if (IntersectRayOBB(p, d, node_obb, t, q, abs_tol))
            {
            // is this node a leaf node?
            unsigned int left_child = getLeftChild(cur_node);
            if (left_child == 0xffffffff)
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
        return (m_left[idx] == 0xffffffff);
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

    DEVICE inline unsigned int getParticleByNode(unsigned int node, unsigned int i) const
        {
        return m_particles[m_leaf_ptr[node] + i];
        }

    DEVICE inline unsigned int getLeafNodePtrByNode(unsigned int node) const
        {
        return m_leaf_ptr[node];
        }

    DEVICE inline unsigned int getParticleByIndex(unsigned int idx) const
        {
        return m_particles[idx];
        }

    DEVICE inline int getNumParticles(unsigned int node) const
        {
        return m_leaf_ptr[node + 1] - m_leaf_ptr[node];
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

#ifdef ENABLE_HIP
    //! Set CUDA memory hints
    void set_memory_hint() const
        {
        m_center.set_memory_hint();
        m_lengths.set_memory_hint();
        m_rotation.set_memory_hint();
        m_mask.set_memory_hint();
        m_is_sphere.set_memory_hint();

        m_left.set_memory_hint();
        m_escape.set_memory_hint();
        m_ancestors.set_memory_hint();

        m_leaf_ptr.set_memory_hint();
        m_leaf_obb_ptr.set_memory_hint();
        m_particles.set_memory_hint();
        }
#endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
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

    //! Determine size of the shared memory allocation
    /*! \param ptr Pointer to increment
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
        {
        m_center.allocate_shared(ptr, available_bytes);
        m_lengths.allocate_shared(ptr, available_bytes);
        m_rotation.allocate_shared(ptr, available_bytes);
        m_mask.allocate_shared(ptr, available_bytes);
        m_is_sphere.allocate_shared(ptr, available_bytes);

        m_left.allocate_shared(ptr, available_bytes);
        m_escape.allocate_shared(ptr, available_bytes);
        m_ancestors.allocate_shared(ptr, available_bytes);

        m_leaf_ptr.allocate_shared(ptr, available_bytes);
        m_leaf_obb_ptr.allocate_shared(ptr, available_bytes);
        m_particles.allocate_shared(ptr, available_bytes);
        }

    //! Get the capacity of leaf nodes
    unsigned int getLeafNodeCapacity() const
        {
        return m_leaf_capacity;
        }

    private:
    ManagedArray<vec3<ShortReal>> m_center;
    ManagedArray<vec3<ShortReal>> m_lengths;
    ManagedArray<quat<ShortReal>> m_rotation;
    ManagedArray<unsigned int> m_mask;
    ManagedArray<unsigned int> m_is_sphere;

    ManagedArray<unsigned int> m_leaf_ptr;     //!< Pointer to leaf node contents
    ManagedArray<unsigned int> m_leaf_obb_ptr; //!< Pointer to leaf node OBBs
    ManagedArray<unsigned int> m_particles;    //!< Stores the leaf nodes' indices

    ManagedArray<unsigned int> m_left;      //!< Left nodes
    ManagedArray<unsigned int> m_escape;    //!< Escape indices
    ManagedArray<unsigned int> m_ancestors; //!< Number of right-most ancestors

    unsigned int m_num_nodes;     //!< Number of nodes in the tree
    unsigned int m_num_leaves;    //!< Number of leaf nodes
    unsigned int m_leaf_capacity; //!< Capacity of OBB leaf nodes
    };

// Tandem stack traversal routines
// from: A Binary Stack Tandem Traversal and an Ancestor Counter Data Structure for GPU friendly
// Bounding Volume Damkjær, Jesper and Erleben, Kenny Proceedings Workshop in Virtual Reality
// Interactions and Physical Simulation "VRIPHYS" (2009)
// https://dx.doi.org/10.2312/PE/vriphys/vriphys09/115-124

//! Compute how many ascents are necessary to reach a non right-most child
/*! \param a_count Ancestor count in tree a
    \param b_count Ancestor count in tree b
    \param Binary stack
    \param a_ascent Number of ascents in a (return variable)
    \param b_ascent Number of ascents in b (return variable)
 */
DEVICE inline void findAscent(unsigned int a_count,
                              unsigned int b_count,
                              unsigned long int& stack,
                              unsigned int& a_ascent,
                              unsigned int& b_ascent)
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
DEVICE inline bool traverseBinaryStack(const GPUTree& a,
                                       const GPUTree& b,
                                       unsigned int& cur_node_a,
                                       unsigned int& cur_node_b,
                                       unsigned long int& stack,
                                       OBB& obb_a,
                                       OBB& obb_b,
                                       const quat<ShortReal>& q,
                                       const vec3<ShortReal>& dr)
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
            bool descend_A = obb_a.getVolume() > obb_b.getVolume() ? !a.isLeaf(cur_node_a)
                                                                   : b.isLeaf(cur_node_b);

            if (descend_A)
                {
                cur_node_a = a.getLeftChild(cur_node_a);
                stack <<= 1; // push A
                }
            else
                {
                cur_node_b = b.getLeftChild(cur_node_b);
                stack <<= 1;
                stack |= 1; // push B
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

//! Traverse a binary hierachy, subject to intersection with a third OBB
/*! Returns true if an intersecting pair of leaf OBB's has been found, where both
 * OBBs intersect with the third OBB
 *
 * \param a First tree
 * \param b Second tree
 * \param cur_node_a Current node in first tree
 * \param cur_node_b Current node in second tree
 * \param a binary stack realized as an integer
 * \param obb_a OBB from first tree corresponding to cur_node_a, in the reference frame of b
 * \param obb_b OBB from second tree corresponding to cur_node_b
 * \param q Rotation that is applied to a's OBBs to bring them into b's reference frame
 * \param dr translation that is applied to a's OBBs to bring them into b's reference frame
 * \param obb_c OBB to test the first two OBB's against, in the reference frame of b
 *
 * This function prefetches OBBs from trees a and b (only when the nodes change)
 * and is supposed to be called from a while-loop:
 *
 * unsigned long int stack = 0;
 * // the third OBB
 * obb_c = ..
 *
 * // load initial OBBs for the two nodes
 * obb_a = ...
 * // transform OBB a into B's frame
 * ...
 * obb_b = ...
 *
 *
 * while (cur_node_a != a.tree.getNumNodes() && cur_node_b != b.tree.getNumNodes())
 *     {
 *     query_node_a = cur_node_a;
 *     query_node_b = cur_node_b;
 *     if (traverseBinaryStack(a, b, cur_node_a, cur_node_b, stack, obb_a, obb_b, obb..))
 *            test_narrow_phase(a, b, query_node_a, query_node_b, ...)
 *     }
 */
DEVICE inline bool traverseBinaryStackIntersection(const GPUTree& a,
                                                   const GPUTree& b,
                                                   unsigned int& cur_node_a,
                                                   unsigned int& cur_node_b,
                                                   unsigned long int& stack,
                                                   OBB& obb_a,
                                                   OBB& obb_b,
                                                   const quat<ShortReal>& q,
                                                   const vec3<ShortReal>& dr,
                                                   const OBB& obb_c)
    {
    bool leaf = false;
    bool ascend = true;

    unsigned int old_a = cur_node_a;
    unsigned int old_b = cur_node_b;

    if (overlap(obb_a, obb_b) && overlap(obb_a, obb_c) && overlap(obb_b, obb_c))
        {
        if (a.isLeaf(cur_node_a) && b.isLeaf(cur_node_b))
            {
            leaf = true;
            }
        else
            {
            // descend into subtree with larger volume first (unless there are no children)
            bool descend_A = obb_a.getVolume() > obb_b.getVolume() ? !a.isLeaf(cur_node_a)
                                                                   : b.isLeaf(cur_node_b);

            if (descend_A)
                {
                cur_node_a = a.getLeftChild(cur_node_a);
                stack <<= 1; // push A
                }
            else
                {
                cur_node_b = b.getLeftChild(cur_node_b);
                stack <<= 1;
                stack |= 1; // push B
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

    } // end namespace detail

    } // end namespace hpmc

    } // end namespace hoomd

#endif // __GPU_TREE_H__
