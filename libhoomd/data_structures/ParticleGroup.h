/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

// Maintainer: joaander

/*! \file ParticleGroup.h
    \brief Declares the ParticleGroup and related classes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include "SystemDefinition.h"

#ifndef __PARTICLE_GROUP_H__
#define __PARTICLE_GROUP_H__

//! Utility class to select particles based on given conditions
/*! \b Overview
    
    In order to flexibly specify the particles that belong to a given ParticleGroup, it will simple take a
    ParticleSelector as a parameter in its constructor. The selector will provide a true/false membership test that will
    be applied to each particle tag, selecting those that belong in the group. As it is specified via a virtual class,
    the group definition can be expanded to include any concievable selection criteria.
    
    <b>Implementation details</b>
    So that an infinite range of selection criteria can be applied (i.e. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a reference to the SystemDefinition on construction, along with
    any parameters to specify the selection criteria. Then, a simple isSelected() test is provided that will acquire the
    needed data and will return true if that particle meets the criteria.
    
    The base class isSelected() method will simply reject all particles. Derived classes will implement specific
    selection semantics.
*/
class ParticleSelector
    {
    public:
        //! constructs a ParticleSelector
        ParticleSelector(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelector() {}

        //! Test if a particle meets the selection criteria
        virtual bool isSelected(unsigned int tag) const;
    protected:
        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience
    };

//! Select particles based on their tag
class ParticleSelectorTag : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorTag(boost::shared_ptr<SystemDefinition> sysdef, unsigned int tag_min, unsigned int tag_max);
        virtual ~ParticleSelectorTag() {}

        //! Test if a particle meets the selection criteria
        virtual bool isSelected(unsigned int tag) const;
    protected:
        unsigned int m_tag_min;     //!< Minimum tag to select
        unsigned int m_tag_max;     //!< Maximum tag to select (inclusive)
    };

//! Select particles based on their type
class ParticleSelectorType : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorType(boost::shared_ptr<SystemDefinition> sysdef, unsigned int typ_min, unsigned int typ_max);
        virtual ~ParticleSelectorType() {}

        //! Test if a particle meets the selection criteria
        virtual bool isSelected(unsigned int tag) const;
    protected:
        unsigned int m_typ_min;     //!< Minimum type to select
        unsigned int m_typ_max;     //!< Maximum type to select (inclusive)
    };

//! Select particles in the space defined by a cuboid
class ParticleSelectorCuboid : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorCuboid(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 min, Scalar3 max);
        virtual ~ParticleSelectorCuboid() {}

        //! Test if a particle meets the selection criteria
        virtual bool isSelected(unsigned int tag) const;
    protected:
        Scalar3 m_min;     //!< Minimum type to select (inclusive)
        Scalar3 m_max;     //!< Maximum type to select (exclusive)
    };

//! Select particles based on their rigid body
class ParticleSelectorRigid : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorRigid(boost::shared_ptr<SystemDefinition> sysdef, bool rigid);
        virtual ~ParticleSelectorRigid() {}

        //! Test if a particle meets the selection criteria
        virtual bool isSelected(unsigned int tag) const;
    protected:
        bool m_rigid;   //!< true if we should select rigid boides, false if we should select non-rigid particles
    };

//! Describes a group of particles
/*! \b Overview
    
    Some computations in HOOMD need to only be performed on certain groups of particles. ParticleGroup facilitates
    that by providing a flexible interface for choosing these groups that can be used by any other class in HOOMD.
    
    The following common use-cases are expected and the design is tuned to make these optimal.
     - Iterate through all particles indices in the group, and the order of iteration doesn't matter, except for 
       performance.
     - Iterate through all particle tags in the group, in a well-defined order that does not change (namely, a sorted
       tag order is required)
     - O(1) test if a particular particle index is in the group

    Membership in the group is determined through a generic ParticleSelector class. See its documentation for details.

    Group membership is determined once at the instantiation of the group. Thus ParticleGroup only supports static
    groups where membership does not change over the course of a simulation. Dynamic groups, if they are needed,
    may require a drastically different design to allow for efficient access.

    In many use-cases, ParticleGroup may be accessed many times within inner loops. Thus, it must not aquire any
    ParticleData arrays within most of the get() calls as the caller must be allowed to leave their ParticleData 
    aquired. Thus, all get() methods must return values from internal cached variables only. Those methods that
    absolutely require the particle data be released before they are called will be documented as such.

    <b>Data Structures and Implementation</b>

    The initial and fundamental data structure in the group is a vector listing all of the particle tags in the group,
    in a sorted tag order. This list can be accessed directly via getMemberTag() to meet the 2nd use case listed above.
    In order to iterate through all particles in the group in a cache-efficient manner, an auxilliary list is stored
    that lists all particle <i>indicies</i> that belong to the group. This list must be updated on every particle sort.
    Thirdly, a dynamic bitset is used to store one bit per particle for efficient O(1) tests if a given particle is in 
    the group.
    
    Finally, the common use case on the GPU using groups will include running one thread per particle in the group.
    For that it needs a list of indices of all the particles in the group. To facilitates this, the list of indices
    in the group will be stored in a GPUArray.

    \ingroup data_structs
*/
class ParticleGroup
    {
    public:
        //! \name Initialization methods
        // @{

        //! Constructs an empty particle group
        ParticleGroup() : m_num_local_members(0) {};

        //! Constructs a particle group of all particles that meet the given selection
        ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleSelector> selector);

        //! Constructs a particle group given a list of tags
        ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& member_tags);

        //! Destructor
        ~ParticleGroup();
        
        // @}
        //! \name Accessor methods
        // @{
                
        //! Get the number of members in the group
        /*! \returns The number of particles that belong to this group
        */
        unsigned int getNumMembers() const
            {
            return (unsigned int)m_member_tags.getNumElements();
            }
            
        //! Get the number of members that are present on the local processor
        /*! \returns The number of particles on the local processor that belong to this group
        */
        unsigned int getNumLocalMembers() const
            {
            return m_num_local_members;
            }

        //! Get a member from the group
        /*! \param i Index from 0 to getNumMembers()-1 of the group member to get
            \returns Tag of the member at index \a i
        */
        unsigned int getMemberTag(unsigned int i) const
            {
            assert(i < getNumMembers());
            ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::read);
            return h_member_tags.data[i];
            }
            
        //! Get a member index from the group
        /*! \param j Value from 0 to getNumMembers()-1 of the group member to get
            \returns Index of the member at position \a j
            \note getMemberTag(j) \b does \b NOT get the tag of the particle with index getMemberIndex(j). These two
                  lists are stored in different orders. Access the ParticleData to convert between tags and indices.
        */
        unsigned int getMemberIndex(unsigned int j) const
            {
            assert(j < getNumLocalMembers());
            ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::read);
            unsigned int idx = h_handle.data[j];
            assert(idx < m_pdata->getN());
            return idx;
            }

        //! Test if a particle index is a member of the group
        /*! \param idx Index of the particle to query (from 0 to the number of partilces in ParticleData -1)
            \returns true if the particle with index \a idx is in the group
        */
        bool isMember(unsigned int idx) const
            {
            ArrayHandle<unsigned char> h_handle(m_is_member, access_location::host, access_mode::read);
            return h_handle.data[idx] == 1;
            }
        
        //! Direct access to the index list
        /*! \returns A GPUArray for directly accessing the index list, intended for use in using groups on the GPU
            \note The caller \b must \b not write to or change the array.
        */
        const GPUArray<unsigned int>& getIndexArray() const
            {
            return m_member_idx;
            }

        // @}
        //! \name Analysis methods
        // @{
        
        //! Compute the total mass of the group
        Scalar getTotalMass() const;
        //! Compute the center of mass of the group
        Scalar3 getCenterOfMass() const;

        // @}
        //! \name Combination methods
        // @{
            
        //! Make a new particle group from a union of two
        static boost::shared_ptr<ParticleGroup> groupUnion(boost::shared_ptr<ParticleGroup> a,
                                                           boost::shared_ptr<ParticleGroup> b);
        //! Make a new particle group from an intersection
        static boost::shared_ptr<ParticleGroup> groupIntersection(boost::shared_ptr<ParticleGroup> a,
                                                                  boost::shared_ptr<ParticleGroup> b);
        //! Make a new particle group from an difference
        static boost::shared_ptr<ParticleGroup> groupDifference(boost::shared_ptr<ParticleGroup> a,
                                                                boost::shared_ptr<ParticleGroup> b);
        
        // @}

    private:
        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition this group is associated with
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data this group is associated with
        GPUArray<unsigned char> m_is_member;            //!< One byte per particle, == 1 if index is a member of the group
        GPUArray<unsigned int> m_member_idx;            //!< List of all particle indices in the group
        boost::signals::connection m_sort_connection;   //!< Connection to the ParticleData sort signal
        boost::signals::connection m_max_particle_num_change_connection; //!< Connection to the max particle number change signal
        boost::signals::connection m_local_particle_num_change_connection; //!< Connection to the local particle number change signal
        GPUArray<unsigned int> m_member_tags;           //!< Lists the tags of the paritcle members
        unsigned int m_num_local_members;               //!< Number of members on the local processor

        //! Helper function to resize array of member tags
        void reallocate();

        //! Helper function to rebuild the index lists afer the particles have been sorted
        void rebuildIndexList();

#ifdef ENABLE_CUDA
        //! Helper function to rebuild the index lists afer the particles have been sorted
        void rebuildIndexListGPU();
#endif

    };

//! Exports the ParticleGroup class to python
void export_ParticleGroup();

#endif

