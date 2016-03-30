#include "OBBTree.h"

#ifndef __GPU_TREE_H__
#define __GPU_TREE_H__

//! Max number of nodes that can be stored in this structure
#define GPU_TREE_MAX_NODES 64

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
        /*! \param tree OBBTree to construct from
         */
        GPUTree(const OBBTree &tree)
            {
            if (tree.getNumNodes() >= GPU_TREE_MAX_NODES)
                {
                throw std::runtime_error("GPUTree: Too many nodes.");
                }

            // load data from AABTree
            for (unsigned int i = 0; i < tree.getNumNodes(); ++i)
                {
                m_left[i] = tree.getNodeLeft(i);
                m_skip[i] = tree.getNodeSkip(i);

                m_center[i] = tree.getNodeOBB(i).getPosition();
                m_rotation[i] = tree.getNodeOBB(i).rotation;
                m_lengths[i] = tree.getNodeOBB(i).lengths;

               for (unsigned int j = 0; j < OBB_NODE_CAPACITY; ++j)
                    {
                    if (j < tree.getNodeNumParticles(i))
                        {
                        m_particles[i*OBB_NODE_CAPACITY+j] = tree.getNodeParticle(i,j);
                        }
                    else
                        {
                        m_particles[i*OBB_NODE_CAPACITY+j] = -1;
                        }
                    }
                }
            m_num_nodes = tree.getNumNodes();

            // update auxillary information for tandem traversal
            updateRCL(0, tree, 0, true, 0, 0);
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
         * \param particles List of particles returned (array of at least OBB_NODE_CAPACITY length), -1 means no particle
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
                    for (unsigned int i = 0; i < OBB_NODE_CAPACITY; i++)
                        particles[i] = m_particles[cur_node*OBB_NODE_CAPACITY+i];
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

        DEVICE inline int getParticle(unsigned int node, unsigned int i) const
            {
            return m_particles[node*OBB_NODE_CAPACITY+i];
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

    protected:
        #ifndef NVCC
        void updateRCL(unsigned int idx, const OBBTree& tree, unsigned int level, bool left,
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
        vec3<OverlapReal> m_center[GPU_TREE_MAX_NODES];
        vec3<OverlapReal> m_lengths[GPU_TREE_MAX_NODES];
        rotmat3<OverlapReal> m_rotation[GPU_TREE_MAX_NODES];

        unsigned int m_level[GPU_TREE_MAX_NODES];              //!< Depth
        bool m_isleft[GPU_TREE_MAX_NODES];                     //!< True if this node is a left node
        unsigned int m_parent[GPU_TREE_MAX_NODES];             //!< Pointer to parent
        unsigned int m_rcl[GPU_TREE_MAX_NODES];                //!< Right child level

        int m_particles[GPU_TREE_MAX_NODES*OBB_NODE_CAPACITY];          //!< Stores the nodes' indices

        unsigned int m_left[GPU_TREE_MAX_NODES];                    //!< Left nodes
        unsigned int m_skip[GPU_TREE_MAX_NODES];                    //!< Skip intervals
        unsigned int m_num_nodes;                                    //!< Number of nodes in the tree
    };

}; // end namespace detail

}; // end namespace hpmc

#endif // __GPU_TREE_H__
