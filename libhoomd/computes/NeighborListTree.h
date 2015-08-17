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

#include "NeighborList.h"
#include "AABBTree.h"
#include <vector>

/*! \file NeighborListTree.h
    \brief Declares the NeighborListTree class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTTREE_H__
#define __NEIGHBORLISTTREE_H__

//! Efficient neighbor list build on the CPU using BVH trees
/*!
 * A bounding volume hierarchy (BVH) tree is a binary search tree. It is constructed from axis-aligned bounding boxes
 * (AABBs). The AABB for a node in the tree encloses all child AABBs. A leaf AABB holds multiple particles. The tree
 * is constructed in a balanced way using a heuristic to minimize AABB volume. We build one tree per particle type,
 * and use point AABBs for the particles. The neighbor list is built by traversing down the tree with an AABB
 * that encloses the pairwise cutoff for the particle. Periodic boundaries are treated by translating the query AABB
 * by all possible image vectors, many of which are trivially rejected for not intersecting the root node.
 *
 * Because one tree is built per type, complications can arise if particles change type "on the fly" during a
 * a simulation. At present, there is no signal for the types of particles changing (only the total number of types).
 * Any class directly modifying the types of particles \b must signal this change to NeighborListTree using
 * notifyParticleSort().
 *
 * \ingroup computes
 */
class NeighborListTree : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListTree(boost::shared_ptr<SystemDefinition> sysdef,
                           Scalar r_cut,
                           Scalar r_buff);

        //! Destructor
        virtual ~NeighborListTree();
            
    protected:
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
        
    private:
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }
        
        //! Notification of a max number of particle change    
        void slotMaxNumChanged()
            {
            m_max_num_changed = true;
            }
            
        //! Notification of a particle sort
        void slotRemapParticles()
            {
            m_remap_particles = true;
            }
        
        //! Notification of a number of types change    
        void slotNumTypesChanged()
            {
            m_type_changed = true;
            }
    
        bool m_box_changed;                                 //!< Flag if box size has changed
        boost::signals2::connection m_boxchange_connection; //!< Connection to the ParticleData box size change signal
        
        bool m_max_num_changed;                             //!< Flag if the particle arrays need to be resized
        boost::signals2::connection m_max_numchange_conn;   //!< Connection to max particle number change signal
        
        bool m_remap_particles;                     //!< Flag if the particles need to remapped (triggered by sort)
        boost::signals2::connection m_sort_conn;    //!< Local connection to the ParticleData sort signal
    
        bool m_type_changed;                                //!< Flag if the number of types has changed
        boost::signals2::connection m_num_type_change_conn; //!< Connection to the ParticleData number of types
    
        // we use stl vectors here because these tree data structures should *never* be
        // accessed on the GPU, they were optimized for the CPU with SIMD support
        std::vector<hpmc::detail::AABBTree>      m_aabb_trees;     //!< Flat array of AABB trees of all types
        std::vector<hpmc::detail::AABB>          m_aabbs;          //!< Flat array of AABBs of all types
        std::vector<unsigned int>  m_num_per_type;   //!< Total number of particles per type
        std::vector<unsigned int>  m_type_head;      //!< Index of first particle of each type, after sorting
        std::vector<unsigned int>  m_map_pid_tree;   //!< Maps the particle id to its tag in tree for sorting

        std::vector< vec3<Scalar> > m_image_list;    //!< List of translation vectors
        unsigned int m_n_images;                //!< The number of image vectors to check
        
        //! Driver for tree configuration
        void setupTree();
        
        //! Maps particles by local id to their id within their type trees
        void mapParticlesByType();
        
        //! Computes the image vectors to query for 
        void updateImageVectors();
        
        //! Driver to build AABB trees
        void buildTree();
        
        //! Traverses AABB trees to compute neighbors
        void traverseTree();
    };

//! Exports NeighborListTree to python
void export_NeighborListTree();

#endif // __NEIGHBORLISTTREE_H__
