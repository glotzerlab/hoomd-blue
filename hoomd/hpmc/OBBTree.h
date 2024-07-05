// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <stack>
#include <vector>

#include "OBB.h"

#ifndef __OBB_TREE_H__
#define __OBB_TREE_H__

/*! \file OBBTree.h
    \brief OBBTree build and query methods
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
/*! \addtogroup overlap
    @{
*/

const unsigned int OBB_INVALID_NODE = 0xffffffff; //!< Invalid node index sentinel

#ifndef __HIPCC__

//! Node in an OBBTree
/*! Stores data for a node in the OBB tree
 */
struct OBBNode
    {
    //! Default constructor
    OBBNode()
        {
        left = right = parent = OBB_INVALID_NODE;
        escape = 0;
        }

    OBB obb;             //!< The box bounding this node's volume
    unsigned int left;   //!< Index of the left child
    unsigned int right;  //!< Index of the right child
    unsigned int parent; //!< Index of the parent node
    unsigned int escape; //!< Index of next node in in-order traversal

    std::vector<unsigned int> particles; //!< Indices of the particles contained in the node
    };

//! OBB Tree
/*! An OBBTree stores a binary tree of OBBs. A leaf node stores up to node_capacity particles by
   index. The bounding box of a leaf node is surrounds all the bounding boxes of its contained
   particles. Internal nodes have OBBs that enclose all of their children. The tree supports the
   following operations:

    - Query  : Search through the tree and build a list of all particles that intersect with the
   query OBB. Runs in O(log N) time
    - buildTree : build an efficiently arranged tree given a complete set of OBBs, one for each
   particle.

    **Implementation details**

    OBBTree stores all nodes in a flat array manged by std::vector. The tree is in *post-order*.
    The nodes store the indices of their left and right children along with their OBB. Nodes
    are allocated as needed with allocate(). With multiple particles per leaf node, the total number
   of internal nodes needed is not known (but can be estimated) until build time.

    For performance, no recursive calls are used. Instead, each function is either turned into a
   loop if it uses tail recursion, or it uses a local stack to traverse the tree. The stack is
   cached between calls to limit the amount of dynamic memory allocation.
*/
class OBBTree
    {
    public:
    //! Construct an OBBTree
    OBBTree() : m_nodes(0), m_num_nodes(0), m_node_capacity(0), m_leaf_capacity(0), m_root(0) { }

    // Destructor
    ~OBBTree()
        {
        if (m_nodes)
            delete[] m_nodes;
        }

    //! Build a tree smartly from a list of OBBs and internal coordinates
    inline void buildTree(OBB* obbs,
                          std::vector<std::vector<vec3<ShortReal>>>& internal_coordinates,
                          ShortReal vertex_radius,
                          unsigned int N,
                          unsigned int leaf_capacity);

    //! Build a tree from a list of OBBs
    inline void buildTree(OBB* obbs, unsigned int N, unsigned int leaf_capacity, bool sphere_tree);

    //! Update the OBB of a particle
    inline void update(unsigned int idx, const OBB& obb);

    //! Get the number of nodes
    inline unsigned int getNumNodes() const
        {
        return m_num_nodes;
        }

    //! Test if a given index is a leaf node
    /*! \param node Index of the node (not the particle) to query
     */
    inline bool isNodeLeaf(unsigned int node) const
        {
        return (m_nodes[node].left == OBB_INVALID_NODE);
        }

    //! Get the OBBNode
    /*! \param node Index of the node (not the particle) to query
     */
    inline const OBBNode& getNode(unsigned int node) const
        {
        return m_nodes[node];
        }

    //! Get the OBB of a given node
    /*! \param node Index of the node (not the particle) to query
     */
    inline const OBB& getNodeOBB(unsigned int node) const
        {
        return (m_nodes[node].obb);
        }

    //! Get the the escape index for a given node
    /*! \param node Index of the node (not the particle) to query
     */
    inline unsigned int getEscapeIndex(unsigned int node) const
        {
        return (m_nodes[node].escape);
        }

    //! Get the left child of a given node
    /*! \param node Index of the node (not the particle) to query
     */
    inline unsigned int getNodeLeft(unsigned int node) const
        {
        return (m_nodes[node].left);
        }

    //! Get the number of particles in a given node
    /*! \param node Index of the node (not the particle) to query
     */
    inline unsigned int getNodeNumParticles(unsigned int node) const
        {
        return ((unsigned int)m_nodes[node].particles.size());
        }

    //! Get the particles in a given node
    /*! \param node Index of the node (not the particle) to query
     */
    inline unsigned int getNodeParticle(unsigned int node, unsigned int j) const
        {
        assert(m_nodes[node].particles.size() > j);
        return (m_nodes[node].particles[j]);
        }

    //! Get the capacity of leaf nodes
    unsigned int getLeafNodeCapacity() const
        {
        return m_leaf_capacity;
        }

    private:
    OBBNode* m_nodes;             //!< The nodes of the tree
    unsigned int m_num_nodes;     //!< Number of nodes
    unsigned int m_node_capacity; //!< Capacity of the nodes array
    unsigned int m_leaf_capacity; //!< Number of particles in leaf nodes
    unsigned int m_root;          //!< Index to the root node of the tree

    //! Initialize the tree to hold N particles
    inline void init(unsigned int N);

    //! Build a node of the tree recursively
    inline unsigned int buildNode(OBB* obbs,
                                  std::vector<std::vector<vec3<ShortReal>>>& internal_coordinates,
                                  std::vector<std::vector<ShortReal>>& vertex_radii,
                                  std::vector<unsigned int>& idx,
                                  unsigned int start,
                                  unsigned int len,
                                  unsigned int parent,
                                  bool sphere_tree);

    //! Allocate a new node
    inline unsigned int allocateNode();

    //! Update the escape index for a node
    inline void updateEscapeIndex(unsigned int idx, unsigned int parent_idx);
    };

/*! \param N Number of particles to allocate space for

    Initialize the tree with room for N particles.
*/
inline void OBBTree::init(unsigned int N)
    {
    // clear the nodes
    m_num_nodes = 0;

    // init the root node to invalid state
    m_root = OBB_INVALID_NODE;
    }

/*! \param obbs List of OBBs for each particle (must be 32-byte aligned)
    \param internal_coordinates List of lists of vertex contents of OBBs
    \param vertex_radius Radius of every vertex
    \param N Number of OBBs in the list

    Builds a balanced tree from a given list of OBBs for each particle. Data in \a obbs will be
   modified during the construction process.
*/
inline void OBBTree::buildTree(OBB* obbs,
                               std::vector<std::vector<vec3<ShortReal>>>& internal_coordinates,
                               ShortReal vertex_radius,
                               unsigned int N,
                               unsigned int leaf_capacity)
    {
    m_leaf_capacity = leaf_capacity;
    init(N);

    std::vector<unsigned int> idx;
    for (unsigned int i = 0; i < N; i++)
        idx.push_back(i);

    std::vector<std::vector<ShortReal>> vertex_radii(N);
    for (unsigned int i = 0; i < N; ++i)
        vertex_radii[i] = std::vector<ShortReal>(internal_coordinates[i].size(), vertex_radius);

    m_root
        = buildNode(obbs, internal_coordinates, vertex_radii, idx, 0, N, OBB_INVALID_NODE, false);
    updateEscapeIndex(m_root, getNumNodes());
    }

/*! \param obbs List of OBBs for each particle (must be 32-byte aligned)
    \param N Number of OBBs in the list

    Builds a balanced tree from a given list of OBBs for each particle. Data in \a obbs will be
   modified during the construction process.
*/
inline void
OBBTree::buildTree(OBB* obbs, unsigned int N, unsigned int leaf_capacity, bool sphere_tree)
    {
    m_leaf_capacity = leaf_capacity;
    init(N);

    std::vector<unsigned int> idx;
    for (unsigned int i = 0; i < N; i++)
        idx.push_back(i);

    // initialize internal coordinates from OBB corners
    std::vector<std::vector<vec3<ShortReal>>> internal_coordinates;
    std::vector<std::vector<ShortReal>> vertex_radii;
    for (unsigned int i = 0; i < N; ++i)
        {
        if (obbs[i].isSphere())
            {
            internal_coordinates.push_back(std::vector<vec3<ShortReal>>(1, obbs[i].getPosition()));

            // all OBB lengths are equal to the radius
            vertex_radii.push_back(std::vector<ShortReal>(1, obbs[i].lengths.x));
            }
        else
            {
            internal_coordinates.push_back(obbs[i].getCorners());
            vertex_radii.push_back(std::vector<ShortReal>(8, 0.0));
            }
        }

    m_root = buildNode(obbs,
                       internal_coordinates,
                       vertex_radii,
                       idx,
                       0,
                       N,
                       OBB_INVALID_NODE,
                       sphere_tree);
    updateEscapeIndex(m_root, getNumNodes());
    }

//! Define a weak ordering on OBB centroid projections
inline bool compare_proj(const std::pair<ShortReal, unsigned int>& lhs,
                         const std::pair<ShortReal, unsigned int>& rhs)
    {
    return lhs.first < rhs.first;
    }

/*! \param obbs List of OBBs
    \param idx List of indices
    \param start Start point in obbs and idx to examine
    \param len Number of obbs to examine
    \param parent Index of the parent node

    buildNode is the main driver of the smart OBB tree build algorithm. Each call produces a node,
   given a set of OBBs. If there are fewer OBBs than fit in a leaf, a leaf is generated. If there
   are too many, the total OBB is computed and split on the largest length axis. The total tree is
   built by recursive splitting.

    The obbs and idx lists are passed in by reference. Each node is given a subrange of the list to
   own (start to start + len). When building the node, it partitions it's subrange into two sides
   (like quick sort).
*/
inline unsigned int
OBBTree::buildNode(OBB* obbs,
                   std::vector<std::vector<vec3<ShortReal>>>& internal_coordinates,
                   std::vector<std::vector<ShortReal>>& vertex_radii,
                   std::vector<unsigned int>& idx,
                   unsigned int start,
                   unsigned int len,
                   unsigned int parent,
                   bool sphere_tree)
    {
    // merge all the OBBs into one, as tightly as possible
    OBB my_obb = obbs[start];

    if (len > 1)
        {
        std::vector<vec3<ShortReal>> merge_internal_coordinates;
        std::vector<ShortReal> merge_vertex_radii;

        for (unsigned int i = start; i < start + len; ++i)
            {
            for (unsigned int j = 0; j < internal_coordinates[i].size(); ++j)
                {
                merge_internal_coordinates.push_back(internal_coordinates[i][j]);
                merge_vertex_radii.push_back(vertex_radii[i][j]);
                }
            }

        // combine masks
        unsigned int mask = 0;

        for (unsigned int i = start; i < start + len; ++i)
            {
            mask |= obbs[i].mask;
            }

        my_obb = compute_obb(merge_internal_coordinates, merge_vertex_radii, sphere_tree);
        my_obb.mask = mask;
        }

    // handle the case of a leaf node creation
    if (len <= m_leaf_capacity)
        {
        unsigned int new_node = allocateNode();
        m_nodes[new_node].obb = my_obb;
        m_nodes[new_node].parent = parent;

        for (unsigned int i = 0; i < len; i++)
            {
            // assign the particle indices into the leaf node
            m_nodes[new_node].particles.push_back(idx[start + i]);
            }

        return new_node;
        }

    // otherwise, we are creating an internal node - allocate an index
    unsigned int my_idx = allocateNode();

    // need to split the list of obbs into two sets for left and right
    unsigned int start_left = 0;
    unsigned int start_right = len;

    rotmat3<ShortReal> my_axes(conj(my_obb.rotation));

    // if there are only 2 obbs, put one on each side
    if (len == 2)
        {
        // nothing to do, already partitioned
        }
    else
        {
        // the x-axis has largest covariance by construction, so split along that axis

        // Object mean
        ShortReal split_proj(0.0);
        for (unsigned int i = 0; i < len; ++i)
            {
            split_proj
                += dot(obbs[start + i].center - my_obb.center, my_axes.row0) / ShortReal(len);
            }

        // split on x direction according to object mean
        for (unsigned int i = 0; i < start_right; i++)
            {
            ShortReal proj = dot(obbs[start + i].center - my_obb.center, my_axes.row0);
            if (proj < split_proj)
                {
                // if on the left side, everything is happy, just continue on
                }
            else
                {
                // if on the right side, need to swap the current obb with the one at start_right-1,
                // subtract one off of start_right to indicate the addition of one to the right side
                // and subtract 1 from i to look at the current index (new obb). This is quick and
                // easy to write, but will randomize indices - might need to look into a stable
                // partitioning algorithm!
                std::swap(obbs[start + i], obbs[start + start_right - 1]);
                std::swap(idx[start + i], idx[start + start_right - 1]);
                std::swap(internal_coordinates[start + i],
                          internal_coordinates[start + start_right - 1]);
                std::swap(vertex_radii[start + i], vertex_radii[start + start_right - 1]);
                start_right--;
                i--;
                }
            }
        }
    // sanity check. The left or right tree may have ended up empty. If so, just borrow one particle
    // from it
    if (start_right == len)
        start_right = len - 1;
    if (start_right == 0)
        start_right = 1;

    // note: calling buildNode has side effects, the m_nodes array may be reallocated. So we need to
    // determine the left and right children, then build our node (can't say m_nodes[my_idx].left =
    // buildNode(...))

    // create nodes in post-order
    unsigned int new_right = buildNode(obbs,
                                       internal_coordinates,
                                       vertex_radii,
                                       idx,
                                       start + start_right,
                                       len - start_right,
                                       my_idx,
                                       sphere_tree);
    unsigned int new_left = buildNode(obbs,
                                      internal_coordinates,
                                      vertex_radii,
                                      idx,
                                      start + start_left,
                                      start_right - start_left,
                                      my_idx,
                                      sphere_tree);

    // now, create the children and connect them up
    m_nodes[my_idx].obb = my_obb;
    m_nodes[my_idx].parent = parent;
    m_nodes[my_idx].left = new_left;
    m_nodes[my_idx].right = new_right;

    return my_idx;
    }

/*! \param idx Index of the node to update

    updateEscapeIndex() pdates the escape index of every node in the tree. The escape index is used
   in the stackless implementation of query. Each node's escape index points to the next node on the
   same level.
*/
inline void OBBTree::updateEscapeIndex(unsigned int idx, unsigned int escape)
    {
    unsigned int left_idx = m_nodes[idx].left;
    unsigned int right_idx = m_nodes[idx].right;

    m_nodes[idx].escape = escape;

    if (isNodeLeaf(idx))
        return;

    updateEscapeIndex(right_idx, escape);
    updateEscapeIndex(left_idx, right_idx);
    }

/*! Allocates a new node in the tree
 */
inline unsigned int OBBTree::allocateNode()
    {
    // grow the memory if needed
    if (m_num_nodes >= m_node_capacity)
        {
        // determine new capacity
        OBBNode* m_new_nodes = NULL;
        unsigned int m_new_node_capacity = m_node_capacity * 2;
        if (m_new_node_capacity == 0)
            m_new_node_capacity = 16;

        // allocate new memory
        m_new_nodes = new OBBNode[m_new_node_capacity];

        // if we have old memory, copy it over
        if (m_nodes != NULL)
            {
            std::copy(m_nodes, m_nodes + m_num_nodes, m_new_nodes);
            delete[] m_nodes;
            }
        m_nodes = m_new_nodes;
        m_node_capacity = m_new_node_capacity;
        }

    m_nodes[m_num_nodes] = OBBNode();
    m_num_nodes++;
    return m_num_nodes - 1;
    }

// end group overlap
/*! @}*/

#endif // __HIPCC__

    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#endif //__OBB_TREE_H__
