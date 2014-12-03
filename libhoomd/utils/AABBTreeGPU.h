/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: mphoward

#include "AABB.h"
#include "AABBTree.h"


#ifndef __AABB_TREE_GPU_H__
#define __AABB_TREE_GPU_H__

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace hpmc
{

namespace detail
{

//! For now, this looks exactly like a AABBNode, need to microbenchmark to see if it can be eliminated
struct AABBNodeGPU
    {
    AABBNodeGPU()
        {
        left = right = parent = INVALID_NODE;
        num_particles = 0;
        skip = 0;
        }
        
    AABBNodeGPU(const AABBNode& node)
        : aabb(node.aabb), left(node.left), right(node.right), parent(node.parent),
        skip(node.skip), num_particles(node.num_particles)
        {
        for (unsigned int cur_part=0; cur_part < NODE_CAPACITY; ++cur_part)
            {
            particles[cur_part] = node.particles[cur_part];
            particle_tags[cur_part] = node.particle_tags[cur_part];
            }
        }
    
    AABB aabb;
    unsigned int left;   //!< Index of the left child
    unsigned int right;  //!< Index of the right child
    unsigned int parent; //!< Index of the parent node
    unsigned int skip;   //!< Number of array indices to skip to get to the next node in an in order traversal
    
    unsigned int particles[NODE_CAPACITY];      //!< Indices of the particles contained in the node
    unsigned int particle_tags[NODE_CAPACITY];  //!< Corresponding particle tags for particles in node
    unsigned int num_particles;                 //!< Number of particles contained in the node
    }__attribute__((aligned(32)));

//! For now, this is just a stripped down AABBTree with HOSTDEVICE added. Will grow with CUDA build calls later
class AABBTreeGPU
    {
    public:
        AABBTreeGPU() : m_nodes(0), m_num_nodes(0) {}
        
        AABBTreeGPU(const AABBTree& tree)
            : m_nodes(0), m_num_nodes(tree.getNumNodes())
            {
            // allocate memory for the aabb nodes
            int retval = posix_memalign((void**)&m_nodes, 32, m_num_nodes*sizeof(AABBNodeGPU));
            if (retval != 0)
                {
                throw runtime_error("Error allocating AABBTreeGPU memory");
                }
                
            for (unsigned int cur_node=0; cur_node < m_num_nodes; ++cur_node)
                {
                m_nodes[cur_node] = AABBNodeGPU(tree.getNode(cur_node));
                }
            }
            
        ~AABBTreeGPU()
            {
            if (m_nodes)
                free(m_nodes);
            }
            
        //! Get the number of nodes
        HOSTDEVICE inline unsigned int getNumNodes() const
            {
            return m_num_nodes;
            }

        //! Test if a given index is a leaf node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline bool isNodeLeaf(unsigned int node) const
            {
            return (m_nodes[node].left == INVALID_NODE);
            }

        //! Get the AABB of a given node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline const AABB& getNodeAABB(unsigned int node) const
            {
            return (m_nodes[node].aabb);
            }

        //! Get the skip of a given node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline unsigned int getNodeSkip(unsigned int node) const
            {
            return (m_nodes[node].skip);
            }

        //! Get the left child of a given node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline unsigned int getNodeLeft(unsigned int node) const
            {
            return (m_nodes[node].left);
            }

        //! Get the number of particles in a given node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline unsigned int getNodeNumParticles(unsigned int node) const
            {
            return (m_nodes[node].num_particles);
            }

        //! Get the particles in a given node
        /*! \param node Index of the node (not the particle) to query
        */
        HOSTDEVICE inline unsigned int getNodeParticle(unsigned int node, unsigned int j) const
            {
            return (m_nodes[node].particles[j]);
            }
            
        //! Get the associate tag for each particle
        /*! \param node Index of the node (not the particle) to query
         *  \param j Local index in particle array for node
         */
        HOSTDEVICE inline unsigned int getNodeParticleTag(unsigned int node, unsigned int j) const
            {
            return (m_nodes[node].particle_tags[j]);
            }   
    private:
        AABBNodeGPU *m_nodes;
        unsigned int m_num_nodes;
    };

}; // end namespace detail

}; // end namespace hpmc

#undef HOSTDEVICE

#endif //__AABB_TREE_GPU_H__
