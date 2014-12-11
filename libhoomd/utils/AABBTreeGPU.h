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
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

namespace detail
{

struct AABBGPU
    {
    DEVICE AABBGPU()
        {
        tag = 0;
        lower.x = lower.y = lower.z = 0.0;
        upper.x = upper.y = upper.z = 0.0;
        }
    
    //! Construct an AABBGPU from a sphere
    /*! \param _position Position of the sphere
        \param radius Radius of the sphere
    */
    DEVICE AABBGPU(const Scalar3& _position, Scalar radius)
        {
        tag = 0;
        lower.x = _position.x - radius;
        lower.y = _position.y - radius;
        lower.z = _position.z - radius;
        upper.x = _position.x + radius;
        upper.y = _position.y + radius;
        upper.z = _position.z + radius;
        }
        
    //! Construct an AABBGPU from a point with a particle tag
    /*! \param _position Position of the point
        \param _tag Global particle tag id
    */
    DEVICE AABBGPU(const vec3<Scalar>& _position, unsigned int _tag)
        {
        tag = _tag;
        lower = vec_to_scalar4(_position,0.0);
        upper = vec_to_scalar4(_position,0.0);
        }
    
    #ifndef NVCC
    //! Construct an AABBGPU from an AABB (which may have SIMD optimization) to plain old data
    /*! \param aabb AABB object
    */
    AABBGPU(const AABB& aabb)
        {
        tag = aabb.tag;
        lower = vec_to_scalar4(aabb.getLower(),0.0);
        upper = vec_to_scalar4(aabb.getUpper(),0.0);
        }
        
    //! assignment operator from AABB
    inline AABBGPU& operator= (const AABB& aabb)
        {
        tag = aabb.tag;
        lower = vec_to_scalar4(aabb.getLower(),0.0);
        upper = vec_to_scalar4(aabb.getUpper(),0.0);
        return *this;
        }
        
    //! Get the AABB's lower point
    vec3<Scalar> getLower() const
        {
        return vec3<Scalar>(lower.x,lower.y,lower.z);
        }

    //! Get the AABB's upper point
    vec3<Scalar> getUpper() const
        {
        return vec3<Scalar>(upper.x,upper.y,upper.z);
        }
    #endif //NVCC
        
    Scalar4 lower;      //!< Lower left corner
    Scalar4 upper;      //!< Upper right corner
    unsigned int tag;   //!< AABB tagging int
    
    } __attribute__((aligned(32)));
    
//! Check if two AABBs overlap
/*! \param a First AABB
    \param b Second AABB
    \returns true when the two AABBs overlap, false otherwise
*/
DEVICE inline bool overlap(const AABBGPU& a, const AABBGPU& b)
    {
    return !(   b.upper.x < a.lower.x
             || b.lower.x > a.upper.x
             || b.upper.y < a.lower.y
             || b.lower.y > a.upper.y
             || b.upper.z < a.lower.z
             || b.lower.z > a.upper.z
            );
    }

//! For now, this looks exactly like a AABBNode, need to microbenchmark to see if it can be eliminated
struct AABBNodeGPU
    {
    DEVICE AABBNodeGPU()
        {
        left = right = parent = INVALID_NODE;
        num_particles = 0;
        skip = 0;
        }
    
    #ifndef NVCC    
    AABBNodeGPU(const AABBNode& node)
        : left(node.left), right(node.right), parent(node.parent),
        skip(node.skip), num_particles(node.num_particles)
        {
        // assign the AABB into an AABBGPU
        aabb = node.aabb;
        
        // copy the data element-wise
        memcpy(particles, node.particles, NODE_CAPACITY*sizeof(unsigned int));
        memcpy(particle_tags, node.particle_tags, NODE_CAPACITY*sizeof(unsigned int));
        }
        
    inline AABBNodeGPU& operator=(const AABBNode& node)
        {
        left = node.left;
        right = node.right;
        parent = node.parent;
        skip = node.skip;
        num_particles = node.num_particles;
        
        // assign the AABB into an AABBGPU
        aabb = node.aabb;
        
        // copy the data element-wise
        memcpy(particles, node.particles, NODE_CAPACITY*sizeof(unsigned int));
        memcpy(particle_tags, node.particle_tags, NODE_CAPACITY*sizeof(unsigned int));
        
        return *this;
        }
    #endif
    
    AABBGPU aabb;
    unsigned int left;   //!< Index of the left child
    unsigned int right;  //!< Index of the right child
    unsigned int parent; //!< Index of the parent node
    unsigned int skip;   //!< Number of array indices to skip to get to the next node in an in order traversal
    
    unsigned int particles[NODE_CAPACITY];      //!< Indices of the particles contained in the node
    unsigned int particle_tags[NODE_CAPACITY];  //!< Corresponding particle tags for particles in node
    unsigned int num_particles;                 //!< Number of particles contained in the node
    }__attribute__((aligned(32)));

//! For now, this is just a stripped down AABBTree with DEVICE added. Will grow with CUDA build calls later
class AABBTreeGPU
    {
    public:
        AABBTreeGPU() : m_num_nodes(0), m_node_head(0) {}
        
        #ifndef NVCC 
//         AABBTreeGPU(const AABBTreeGPU& tree)
//             {
//             copyTree<AABBTreeGPU>(tree);
//             }
//             
//         inline AABBTreeGPU& operator= (const AABBTreeGPU& tree)
//             {
//             copyTree<AABBTreeGPU>(tree);
//             return *this;
//             }
               
        AABBTreeGPU(const AABBTree& tree)
            {
            m_num_nodes = tree.getNumNodes();
            m_node_head = 0;
            }
            
        inline AABBTreeGPU& operator= (const AABBTree& tree)
            {
            m_num_nodes = tree.getNumNodes();
            m_node_head = 0;
            return *this;
            }
        #endif
            
        ~AABBTreeGPU()
            {
//             if (m_nodes)
//                 free(m_nodes);
            }
            
        //! Get the number of nodes
        DEVICE inline unsigned int getNumNodes() const
            {
            return m_num_nodes;
            }
        
        DEVICE inline unsigned int getNodeHead() const
            {
            return m_node_head;
            }
        
        DEVICE inline void setNodeHead(unsigned int node_head)
            {
            m_node_head = node_head;
            }

//         //! Test if a given index is a leaf node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline bool isNodeLeaf(unsigned int node) const
//             {
//             return (m_nodes[node].left == INVALID_NODE);
//             }
//             
//         //! Get the AABBNode
//         /*! \param node Index of the node (not the particle) to query
//          */
//         DEVICE inline const AABBNodeGPU& getNode(unsigned int node) const
//             {
//             return m_nodes[node];
//             }
//             
//         //! Get the AABB of a given node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline const AABBGPU& getNodeAABB(unsigned int node) const
//             {
//             return (m_nodes[node].aabb);
//             }
// 
//         //! Get the skip of a given node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline unsigned int getNodeSkip(unsigned int node) const
//             {
//             return (m_nodes[node].skip);
//             }
// 
//         //! Get the left child of a given node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline unsigned int getNodeLeft(unsigned int node) const
//             {
//             return (m_nodes[node].left);
//             }
// 
//         //! Get the number of particles in a given node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline unsigned int getNodeNumParticles(unsigned int node) const
//             {
//             return (m_nodes[node].num_particles);
//             }
// 
//         //! Get the particles in a given node
//         /*! \param node Index of the node (not the particle) to query
//         */
//         DEVICE inline unsigned int getNodeParticle(unsigned int node, unsigned int j) const
//             {
//             return (m_nodes[node].particles[j]);
//             }
//             
//         //! Get the associate tag for each particle
//         /*! \param node Index of the node (not the particle) to query
//          *  \param j Local index in particle array for node
//          */
//         DEVICE inline unsigned int getNodeParticleTag(unsigned int node, unsigned int j) const
//             {
//             return (m_nodes[node].particle_tags[j]);
//             }   
    private:
        unsigned int m_num_nodes;
        unsigned int m_node_head;
        
        //! Copies tree data from CPU or GPU tree
//         #ifndef NVCC
//         template<class T>
//         inline void copyTree(const T& tree)
//             {
//             if (m_nodes)
//                 free(m_nodes);
//                 
//             m_num_nodes = tree.getNumNodes();
//             
//             // allocate memory for the aabb nodes
//             int retval = posix_memalign((void**)&m_nodes, 32, m_num_nodes*sizeof(AABBNodeGPU));
//             if (retval != 0)
//                 {
//                 throw runtime_error("Error allocating memory to copy AABBTreeGPU");
//                 }
//             
//             // assign into AABBNodeGPU
//             for (unsigned int cur_node=0; cur_node < m_num_nodes; ++cur_node)
//                 {
//                 m_nodes[cur_node] = tree.getNode(cur_node);
//                 }
//             }
//         #endif
    };

}; // end namespace detail

}; // end namespace hpmc

#undef DEVICE

#endif //__AABB_TREE_GPU_H__
