#include "hoomd/AABBTree.h"

#ifndef __GPU_TREE_H__
#define __GPU_TREE_H__

//! Max number of nodes that can be stored in this structure
#define GPU_TREE_MAX_NODES 16

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

//! Adapter class to AABTree for query on the GPU
class GPUTree
    {
    public:
        //! Empty constructor
        GPUTree()
            : m_num_nodes(0)
            { }

        #ifndef NVCC
        //! Constructor
        /*! \param tree AABBTree to construct from
            \param leaf_aabb AABBs of leaf nodes
         */
        GPUTree(const AABBTree &tree, const AABB *leaf_aabb)
            {
            if (tree.getNumNodes() >= GPU_TREE_MAX_NODES)
                {
                throw std::runtime_error("GPUTree: Too many nodes.");
                }

            // load data from AABTree
            for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
                {
                m_left[i] = tree.getNodeLeft(i);;
                m_skip[i] = tree.getNodeSkip(i);

                m_lower[i] = tree.getNodeAABB(i).getLower();
                m_upper[i] = tree.getNodeAABB(i).getUpper();

               for (unsigned int j = 0; j < NODE_CAPACITY; ++j)
                    {
                    if (j < tree.getNodeNumParticles(i))
                        {
                        m_particles[i*NODE_CAPACITY+j] = tree.getNodeParticle(i,j);
                        m_leaf_aabb[i*NODE_CAPACITY+j] = leaf_aabb[tree.getNodeParticle(i,j)];
                        }
                    else
                        {
                        m_particles[i*NODE_CAPACITY+j] = -1;
                        }
                    }
                }
            m_num_nodes = tree.getNumNodes();
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
         * \param aabb Query bounding box
         * \param cur_node If 0, start a new tree traversal, otherwise use stored value from previous call
         * \param particles List of particles returned (array of at least NODE_CAPACITY length), -1 means no particle
         * \returns true if the current node overlaps and is a leaf node
         */
        DEVICE inline bool queryNode(const AABB& aabb, unsigned int &cur_node, int *particles) const
            {
            AABB node_aabb(m_lower[cur_node],m_upper[cur_node]);

            bool leaf = false;
            if (overlap(node_aabb, aabb))
                {
                // is this node a leaf node?
                if (m_left[cur_node] == INVALID_NODE)
                    {
                    for (unsigned int i = 0; i < NODE_CAPACITY; i++)
                        particles[i] = m_particles[cur_node*NODE_CAPACITY+i];
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

        //! return a leaf AABB
        DEVICE const AABB& getLeafAABB(unsigned int cur_node, unsigned int i) const
            {
            return m_leaf_aabb[cur_node*NODE_CAPACITY+i];
            }

    private:
        vec3<Scalar> m_lower[GPU_TREE_MAX_NODES];              //!< Lower box boundaries
        vec3<Scalar> m_upper[GPU_TREE_MAX_NODES];              //!< Upper box boundaries
        int m_particles[GPU_TREE_MAX_NODES*NODE_CAPACITY];          //!< Stores the nodes' indices
        AABB m_leaf_aabb[GPU_TREE_MAX_NODES*NODE_CAPACITY];    //!< Stores leaf AABB's for fine grained overlap check

        unsigned int m_left[GPU_TREE_MAX_NODES];                    //!< Left nodes
        unsigned int m_skip[GPU_TREE_MAX_NODES];                    //!< Skip intervals
        unsigned int m_num_nodes;                                    //!< Number of nodes in the tree
    };

}; // end namespace detail

}; // end namespace hpmc

#endif // __GPU_TREE_H__
