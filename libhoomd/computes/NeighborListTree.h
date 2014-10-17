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

using namespace hpmc::detail;

/*! \file NeighborListTree.h
    \brief Declares the NeighborListTree class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTTREE_H__
#define __NEIGHBORLISTTREE_H__

//! Efficient neighbor list build on the CPU
/*! Implements the O(N) neighbor list build on the CPU using a cell list.

    \ingroup computes
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
        
        //! Notification of a box size change
        void slotBoxChanged()
            {
            m_box_changed = true;
            }
            
    protected:
        GPUArray<AABBTree>      m_aabb_trees;           //!< Array of AABB trees
        GPUArray<AABB>          m_aabbs;                //!< Array of AABBs
        GPUArray<unsigned int>  m_num_per_type;         //!< Number of particles per type
        GPUArray<unsigned int>  m_type_head;            //!< Head list to each particle type
        GPUArray<unsigned int>  m_map_p_global_tree;    //!< maps global ids to tag in tree
        GPUArray<unsigned int>  m_map_p_tree_global;    //!< maps tree ids to global ids

        GPUArray< vec3<Scalar> >       m_image_list;           //!< list of translation vectors

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
        virtual void buildTree();
        virtual void traverseTree();
        virtual void getNumPerType();
        
        void allocateTree(unsigned int n_local);
        
        bool m_box_changed;
        boost::signals2::connection m_boxchange_connection;   //!< Connection to the ParticleData box size change signal
        
    private:
        unsigned int m_max_n_local; //!< Maximum number of particles locally
    };

//! Exports NeighborListTree to python
void export_NeighborListTree();

#endif // __NEIGHBORLISTTREE_H__
