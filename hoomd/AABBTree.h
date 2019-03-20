// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "HOOMDMath.h"
#include "VectorMath.h"
#include <vector>
#include <stack>

#include "AABB.h"

#ifndef __AABB_TREE_H__
#define __AABB_TREE_H__

/*! \file AABBTree.h
    \brief AABBTree build and query methods
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

/*! \addtogroup overlap
    @{
*/

const unsigned int NODE_CAPACITY = 16;           //!< Maximum number of particles in a node
const unsigned int INVALID_NODE = 0xffffffff;   //!< Invalid node index sentinel

#ifndef NVCC

//! Node in an AABBTree
/*! Stores data for a node in the AABB tree
*/
struct PYBIND11_EXPORT AABBNode
    {
    //! Default constructor
    AABBNode()
        {
        left = right = parent = INVALID_NODE;
        num_particles = 0;
        skip = 0;
        }

    AABB aabb;           //!< The box bounding this node's volume
    unsigned int left;   //!< Index of the left child
    unsigned int right;  //!< Index of the right child
    unsigned int parent; //!< Index of the parent node
    unsigned int skip;   //!< Number of array indices to skip to get to the next node in an in order traversal

    unsigned int particles[NODE_CAPACITY];      //!< Indices of the particles contained in the node
    unsigned int particle_tags[NODE_CAPACITY];  //!< Corresponding particle tags for particles in node
    unsigned int num_particles;                 //!< Number of particles contained in the node
    } __attribute__((aligned(32)));

//! AABB Tree
/*! An AABBTree stores a binary tree of AABBs. A leaf node stores up to NODE_CAPACITY particles by index. The bounding
    box of a leaf node surrounds all the bounding boxes of its contained particles. Internal nodes have AABBs that
    enclose all of their children. The tree supports the following operations:

    - Query  : Search through the tree and build a list of all particles that intersect with the query AABB. Runs in
               O(log N) time
    - Update : Update the AABB for a selected particle. Updating works well only for small movements as the tree
               topology is left unchanged. Runs in O(log N) time. AABBs are not saved for all particles, so
               an update will only increase the volume of nodes. The tree should be rebuilt periodically instead of
               continually updated.
    - buildTree : build an efficiently arranged tree given a complete set of AABBs, one for each particle.

    **Implementation details**

    AABBTree stores all nodes in a flat array managed by std::vector. To easily locate particle leaf nodes for update,
    a reverse mapping is stored to locate the leaf node containing a particle. m_root tracks the index of the root node
    as the tree is built. The nodes store the indices of their left and right children along with their AABB. Nodes
    are allocated as needed with allocate(). With multiple particles per leaf node, the total number of internal nodes
    needed is not known (but can be estimated) until build time.

    For performance, no recursive calls are used. Instead, each function is either turned into a loop if it uses
    tail recursion, or it uses a local stack to traverse the tree. The stack is cached between calls to limit
    the amount of dynamic memory allocation.
*/
class PYBIND11_EXPORT AABBTree
    {
    public:
        //! Construct an AABBTree
        AABBTree()
            : m_nodes(0), m_num_nodes(0), m_node_capacity(0), m_root(0)
            {
            }

        // Destructor
        ~AABBTree()
            {
            if (m_nodes)
                free(m_nodes);
            }

        //! Copy constructor
        AABBTree(const AABBTree& from)
            {
            m_num_nodes = from.m_num_nodes;
            m_node_capacity = from.m_node_capacity;
            m_root = from.m_root;
            m_mapping = from.m_mapping;

            m_nodes = NULL;

            if (from.m_nodes)
                {
                // allocate memory
                int retval = posix_memalign((void**)&m_nodes, 32, m_node_capacity*sizeof(AABBNode));
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating AABBTree memory");
                    }

                // copy over data
                std::copy(from.m_nodes, from.m_nodes + m_num_nodes, m_nodes);
                }
            }

        //! Copy assignment
        AABBTree& operator=(const AABBTree& from)
            {
            m_num_nodes = from.m_num_nodes;
            m_node_capacity = from.m_node_capacity;
            m_root = from.m_root;
            m_mapping = from.m_mapping;

            if (m_nodes)
                free(m_nodes);

            m_nodes = NULL;

            if (from.m_nodes)
                {
                // allocate memory
                int retval = posix_memalign((void**)&m_nodes, 32, m_node_capacity*sizeof(AABBNode));
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating AABBTree memory");
                    }

                // copy over data
                std::copy(from.m_nodes, from.m_nodes + m_num_nodes, m_nodes);
                }
            return *this;
            }

        //! Build a tree smartly from a list of AABBs
        inline void buildTree(AABB *aabbs, unsigned int N);

        //! Find all particles that overlap with the query AABB
        inline unsigned int query(std::vector<unsigned int>& hits, const AABB& aabb) const;

        //! Update the AABB of a particle
        inline void update(unsigned int idx, const AABB& aabb);

        //! Get the height of a given particle's leaf node
        inline unsigned int height(unsigned int idx);

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
            return (m_nodes[node].left == INVALID_NODE);
            }

        //! Get the AABBNode
        /*! \param node Index of the node (not the particle) to query
         */
        inline const AABBNode& getNode(unsigned int node) const
            {
            return m_nodes[node];
            }

        //! Get the AABB of a given node
        /*! \param node Index of the node (not the particle) to query
        */
        inline const AABB& getNodeAABB(unsigned int node) const
            {
            return (m_nodes[node].aabb);
            }

        //! Get the skip of a given node
        /*! \param node Index of the node (not the particle) to query
        */
        inline unsigned int getNodeSkip(unsigned int node) const
            {
            return (m_nodes[node].skip);
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
            return (m_nodes[node].num_particles);
            }

        //! Get the particles in a given node
        /*! \param node Index of the node (not the particle) to query
        */
        inline unsigned int getNodeParticle(unsigned int node, unsigned int j) const
            {
            return (m_nodes[node].particles[j]);
            }

        //! Get the associate tag for each particle
        /*! \param node Index of the node (not the particle) to query
         *  \param j Local index in particle array for node
         */
        inline unsigned int getNodeParticleTag(unsigned int node, unsigned int j) const
            {
            return (m_nodes[node].particle_tags[j]);
            }
    private:
        AABBNode *m_nodes;                  //!< The nodes of the tree
        unsigned int m_num_nodes;           //!< Number of nodes
        unsigned int m_node_capacity;       //!< Capacity of the nodes array
        unsigned int m_root;                //!< Index to the root node of the tree
        std::vector<unsigned int> m_mapping;//!< Reverse mapping to find node given a particle index

        //! Initialize the tree to hold N particles
        inline void init(unsigned int N);

        //! Build a node of the tree recursively
        inline unsigned int buildNode(AABB *aabbs, std::vector<unsigned int>& idx, unsigned int start, unsigned int len, unsigned int parent);

        //! Allocate a new node
        inline unsigned int allocateNode();

        //! Update the skip value for a node
        inline unsigned int updateSkip(unsigned int idx);
    };


/*! \param N Number of particles to allocate space for

    Initialize the tree with room for N particles.
*/
inline void AABBTree::init(unsigned int N)
    {
    // clear the nodes
    m_num_nodes = 0;

    // init the root node and mapping to invalid states
    m_root = INVALID_NODE;
    m_mapping.resize(N);

    for (unsigned int i = 0; i < N; i++)
        m_mapping[i] = INVALID_NODE;
    }

/*! \param hits Output vector of positive hits.
    \param aabb The AABB to query
    \returns the number of box overlap checks made during the recursion

    The *hits* vector is not cleared, elements are only added with push_back. query() traverses the tree and finds all
    of the leaf nodes that intersect *aabb*. The index of each intersecting leaf node is added to the hits vector.
*/
inline unsigned int AABBTree::query(std::vector<unsigned int>& hits, const AABB& aabb) const
    {
    unsigned int box_overlap_counts = 0;

    // avoid pointer indirection overhead of std::vector
    AABBNode* nodes = &m_nodes[0];

    // stackless search
    for (unsigned int current_node_idx = 0; current_node_idx < m_num_nodes; current_node_idx++)
        {
        // cache current node pointer
        const AABBNode& current_node = nodes[current_node_idx];

        box_overlap_counts++;
        if (overlap(current_node.aabb, aabb))
            {
            if (current_node.left == INVALID_NODE)
                {
                for (unsigned int i = 0; i < current_node.num_particles; i++)
                    hits.push_back(current_node.particles[i]);
                }
            }
        else
            {
            // skip ahead
            current_node_idx += current_node.skip;
            }
        }

    return box_overlap_counts;
    }


/*! \param idx Particle index to update
    \param aabb New AABB for particle *idx*

    Update the node for particle *idx* and its parent nodes to reflect a new position and/or extents. update() does not
    change the tree topology, so it is best for slight changes.
*/
inline void AABBTree::update(unsigned int idx, const AABB& aabb)
    {
    assert(idx < m_mapping.size());

    // find the node this particle is in
    unsigned int node_idx = m_mapping[idx];
    assert(node_idx != INVALID_NODE);

    // grow its AABB if needed
    if (!contains(m_nodes[node_idx].aabb, aabb))
        {
        m_nodes[node_idx].aabb = merge(m_nodes[node_idx].aabb, aabb);

        // update all parent node AABBs
        unsigned int current_node = m_nodes[node_idx].parent;
        while (current_node != INVALID_NODE)
            {
            unsigned int left_idx = m_nodes[current_node].left;
            unsigned int right_idx = m_nodes[current_node].right;

            m_nodes[current_node].aabb = merge(m_nodes[left_idx].aabb, m_nodes[right_idx].aabb);
            current_node = m_nodes[current_node].parent;
            }
        }
    }

/*! \param idx Particle to get height for
    \returns Height of the node
*/
inline unsigned int AABBTree::height(unsigned int idx)
    {
    assert(idx < m_mapping.size());

    // find the node this particle is in
    unsigned int node_idx = m_mapping[idx];

    // handle invalid nodes
    if (node_idx == INVALID_NODE)
        return 0;

    // follow the parent pointers up and count the steps
    unsigned int height = 1;

    unsigned int current_node = m_nodes[node_idx].parent;
    while (current_node != INVALID_NODE)
        {
        current_node = m_nodes[current_node].parent;
        height += 1;
        }

    return height;
    }


/*! \param aabbs List of AABBs for each particle (must be 32-byte aligned)
    \param N Number of AABBs in the list

    Builds a balanced tree from a given list of AABBs for each particle. Data in \a aabbs will be modified during
    the construction process.
*/
inline void AABBTree::buildTree(AABB *aabbs, unsigned int N)
    {
    init(N);

    std::vector<unsigned int> idx;
    for (unsigned int i = 0; i < N; i++)
        idx.push_back(i);

    m_root = buildNode(aabbs, idx, 0, N, INVALID_NODE);
    updateSkip(m_root);
    }

/*! \param aabbs List of AABBs
    \param idx List of indices
    \param start Start point in aabbs and idx to examine
    \param len Number of aabbs to examine
    \param parent Index of the parent node

    buildNode is the main driver of the smart AABB tree build algorithm. Each call produces a node, given a set of
    AABBs. If there are fewer AABBs than fit in a leaf, a leaf is generated. If there are too many, the total AABB
    is computed and split on the largest length axis. The total tree is built by recursive splitting.

    The aabbs and idx lists are passed in by reference. Each node is given a subrange of the list to own (start to
    start + len). When building the node, it partitions its subrange into two sides (like quick sort).
*/
inline unsigned int AABBTree::buildNode(AABB *aabbs,
                                        std::vector<unsigned int>& idx,
                                        unsigned int start,
                                        unsigned int len,
                                        unsigned int parent)
    {
    // merge all the AABBs into one
    AABB my_aabb = aabbs[start];
    for (unsigned int i = 1; i < len; i++)
        {
        my_aabb = merge(my_aabb, aabbs[start+i]);
        }
    vec3<Scalar> my_radius = my_aabb.getUpper() - my_aabb.getLower();

    // handle the case of a leaf node creation
    if (len <= NODE_CAPACITY)
        {
        unsigned int new_node = allocateNode();
        m_nodes[new_node].aabb = my_aabb;
        m_nodes[new_node].parent = parent;
        m_nodes[new_node].num_particles = len;

        for (unsigned int i = 0; i < len; i++)
            {
            // assign the particle indices into the leaf node
            m_nodes[new_node].particles[i] = idx[start+i];
            m_nodes[new_node].particle_tags[i] = aabbs[start+i].tag;

            // assign the reverse mapping from particle indices to leaf node indices
            m_mapping[idx[start+i]] = new_node;
            }

        return new_node;
        }

    // otherwise, we are creating an internal node - allocate an index
    unsigned int my_idx = allocateNode();

    // need to split the list of aabbs into two sets for left and right
    unsigned int start_left = 0;
    unsigned int start_right = len;

    // if there are only 2 aabbs, put one on each side
    if (len == 2)
        {
        // nothing to do, already partitioned
        }
    else
        {
        // otherwise, we need to split them based on a heuristic. split the longest dimension in half
        if (my_radius.x > my_radius.y && my_radius.x > my_radius.z)
            {
            // split on x direction
            for (unsigned int i = 0; i < start_right; i++)
                {
                if (aabbs[start+i].getPosition().x < my_aabb.getPosition().x)
                    {
                    // if on the left side, everything is happy, just continue on
                    }
                else
                    {
                    // if on the right side, need to swap the current aabb with the one at start_right-1, subtract
                    // one off of start_right to indicate the addition of one to the right side and subtract 1
                    // from i to look at the current index (new aabb). This is quick and easy to write, but will
                    // randomize indices - might need to look into a stable partitioning algorithm!
                    std::swap(aabbs[start+i], aabbs[start+start_right-1]);
                    std::swap(idx[start+i], idx[start+start_right-1]);
                    start_right--;
                    i--;
                    }
                }
            }
        else if (my_radius.y > my_radius.z)
            {
            // split on y direction
            for (unsigned int i = 0; i < start_right; i++)
                {
                if (aabbs[start+i].getPosition().y < my_aabb.getPosition().y)
                    {
                    // if on the left side, everything is happy, just continue on
                    }
                else
                    {
                    // if on the right side, need to swap the current aabb with the one at start_right-1, subtract
                    // one off of start_right to indicate the addition of one to the right side and subtract 1
                    // from i to look at the current index (new aabb). This is quick and easy to write, but will
                    // randomize indices - might need to look into a stable partitioning algorithm!
                    std::swap(aabbs[start+i], aabbs[start+start_right-1]);
                    std::swap(idx[start+i], idx[start+start_right-1]);
                    start_right--;
                    i--;
                    }
                }
            }
        else
            {
            // split on z direction
            for (unsigned int i = 0; i < start_right; i++)
                {
                if (aabbs[start+i].getPosition().z < my_aabb.getPosition().z)
                    {
                    // if on the left side, everything is happy, just continue on
                    }
                else
                    {
                    // if on the right side, need to swap the current aabb with the one at start_right-1, subtract
                    // one off of start_right to indicate the addition of one to the right side and subtract 1
                    // from i to look at the current index (new aabb). This is quick and easy to write, but will
                    // randomize indices - might need to look into a stable partitioning algorithm!
                    std::swap(aabbs[start+i], aabbs[start+start_right-1]);
                    std::swap(idx[start+i], idx[start+start_right-1]);
                    start_right--;
                    i--;
                    }
                }
            }
        }

    // sanity check. The left or right tree may have ended up empty. If so, just borrow one particle from it
    if (start_right == len)
        start_right = len-1;
    if (start_right == 0)
        start_right = 1;

    // note: calling buildNode has side effects, the m_nodes array may be reallocated. So we need to determine the left
    // and right children, then build our node (can't say m_nodes[my_idx].left = buildNode(...))
    unsigned int new_left = buildNode(aabbs, idx, start+start_left, start_right-start_left, my_idx);
    unsigned int new_right = buildNode(aabbs, idx, start+start_right, len-start_right, my_idx);

    // now, create the children and connect them up
    m_nodes[my_idx].aabb = my_aabb;
    m_nodes[my_idx].parent = parent;
    m_nodes[my_idx].left = new_left;
    m_nodes[my_idx].right = new_right;

    return my_idx;
    }

/*! \param idx Index of the node to update

    updateSkip() updates the skip field of every node in the tree. The skip field is used in the stackless
    implementation of query. Each node's skip field lists the number of nodes that are children to this node. Because
    of the order in which nodes are built in buildNode(), this number is the number of elements to skip in a search
    if a box-box test does not overlap.
*/
inline unsigned int AABBTree::updateSkip(unsigned int idx)
    {
    // leaf nodes have no nodes under them
    if (isNodeLeaf(idx))
        {
        return 1;
        }
    else
        {
        // node idx needs to skip all the nodes underneath it (determined recursively)
        unsigned int left_idx = m_nodes[idx].left;
        unsigned int right_idx = m_nodes[idx].right;

        unsigned int skip = updateSkip(left_idx) + updateSkip(right_idx);
        m_nodes[idx].skip = skip;
        return skip + 1;
        }
    }

/*! Allocates a new node in the tree
*/
inline unsigned int AABBTree::allocateNode()
    {
    // grow the memory if needed
    if (m_num_nodes >= m_node_capacity)
        {
        // determine new capacity
        AABBNode *m_new_nodes = NULL;
        unsigned int m_new_node_capacity = m_node_capacity*2;
        if (m_new_node_capacity == 0)
            m_new_node_capacity = 16;

        // allocate new memory
        int retval = posix_memalign((void**)&m_new_nodes, 32, m_new_node_capacity*sizeof(AABBNode));
        if (retval != 0)
            {
            throw std::runtime_error("Error allocating AABBTree memory");
            }

        // if we have old memory, copy it over
        if (m_nodes != NULL)
            {
            memcpy((void *)m_new_nodes, (void *)m_nodes, sizeof(AABBNode)*m_num_nodes);
            free(m_nodes);
            }
        m_nodes = m_new_nodes;
        m_node_capacity = m_new_node_capacity;
        }

    m_nodes[m_num_nodes] = AABBNode();
    m_num_nodes++;
    return m_num_nodes-1;
    }

// end group overlap
/*! @}*/

#endif // NVCC

}; // end namespace detail

}; // end namespace hpmc

#endif //__AABB_TREE_H__
