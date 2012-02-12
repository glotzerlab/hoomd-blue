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

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include "SystemDefinition.h"
#include "ParticleSelectorRules.h"

#ifdef ENABLE_CUDA
#include "AllParticleSelectors.cuh"
#endif

#ifndef __PARTICLE_GROUP_H__
#define __PARTICLE_GROUP_H__

//! Utility class to select particles based on given conditions
/*! \b Overview
    
    In order to flexibly specify the particles that belong to a given ParticleGroup, it will take a
    ParticleSelector as a parameter in its constructor. The selector will return a list of selected global tags
    that are owned by the local ParticleData. This list is constructed by applying a selection criterium to every particle.

    Every ParticleGroup has to have a ParticleSelector. This requirement arises since the local members of a group
    may change if particles are inserted or deleted into the simulation domain.
    In such an event, the selector is called to re-determine the tags local to this processor that need to be included.

    If set operations on groups are performed, such as intersection, union or difference, these really
    operate on the ParticleSelectors. In fact, the operations are implemented as selectors themselves,
    which internally store the ParticleSelector classes of the ParticleGroups they take as an argument.
    
    <b>Implementation details</b>
    As the ParticleSelector is specified via a virtual class the group definition can be expanded to include any
    concievable selection criteria.

    So that an infinite range of selection criteria can be applied (i.e. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a reference to the SystemDefinition on construction, along with
    any parameters to specify the selection criteria.

    The ParticleSelector base class only provides an interface. An important implementation is given by the
    rule-based particle ParticleSelectorRule<>, it is a template class that takes a rule as a parameter. This rule
    can be efficiently evaluated on the CPU or on the GPU to construct the tag list.
    
    ParticleSelector provides a purely virtual internal method getMemberTags()
    that is used to generate the list of included tags of local particles. This method is called from within a
    ParticleGroup, if the latter has detected a change in the particle data arrays (e.g.
    insertion or deletion of particles).

    For convenience and compatibility with older versions of HOOMD, instantiations of the ParticleSelectorRule<>
    template class are provided by deriving from it, e.g. there is a ParticleSelectorTag that instantiates a
    ParticleSelectorRule<GlobalTagRule>.
*/
class ParticleSelector
    {
    public:
        //! constructs a ParticleSelector
        ParticleSelector() {}

        virtual ~ParticleSelector() {}

        //! Get the list of selected tags
        /*! \param member_tags GPU array to store the member tags in
         * \returns number of local particles included
         * \pre member_tags must be allocated and of sufficient size to accomodate
         *       all local members of the group (i.e.
         *       the current maximum number of particles returned by ParticleData::getMaxN() )
        */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags) = 0;
    };

//! Implementation of a particle selector that applies a selection rule
/* The selection rule is itself implemented as an evaluator template, to avoid code duplication
   and selection of particles on the GPU.
 */
template< class T >
class ParticleSelectorRule : public ParticleSelector
    {
    public:
        //! constructs a particle selector
        //! \param sysdef the system definition to build the particle group from
        ParticleSelectorRule(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelectorRule() {}

        //! set the parameters of the selector
        /*! \param params the selector parameters
        */
        void setParams(typename T::param_type params);

        //! Get the list of selected tags
        /*! \param member_tags GPU array to store the member tags in
         * \return the number of local particles included
         * \pre  member_tags must be allocated and of sufficient size to accomodate
         *       all local members of the group (i.e.
         *       the current maximum number of particles returned by ParticleData::getMaxN() )
        */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags);

    protected:
        typename T::param_type m_params; //!< rule parameters

        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience

    };

//! Constructor
template< class T >
ParticleSelectorRule<T>::ParticleSelectorRule(boost::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData())
    {
    }

//! Loop over particles and determine those that match the selection criterium
template< class T >
unsigned int ParticleSelectorRule<T>::getMemberTags(const GPUArray<unsigned int>& member_tags)
    {
    assert(member_tags.getNumElements() >= m_pdata->getN());

    unsigned int num_members = 0;
    // instantiate the selection rule
    T rule(m_params);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_member_tags(member_tags, access_location::host, access_mode::overwrite);

    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        // add the tag to the list if it matches the selection
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        unsigned int body = h_body.data[idx];
        unsigned int global_tag = h_global_tag.data[idx];

        if (rule.isSelected(global_tag, body, type))
            h_member_tags.data[num_members++] = global_tag;
        }

    return num_members;
    }

//! Set the parameters for the rule
template< class T >
void ParticleSelectorRule<T>::setParams(typename T::param_type params)
    {
    m_params = params;
    }


//! ParticleSelector to select particles based on the tag rule
class ParticleSelectorTag : public ParticleSelectorRule<GlobalTagRule>
    {
    public:
        //! Initializes the selector
        ParticleSelectorTag(boost::shared_ptr<SystemDefinition> sysdef, unsigned int tag_min, unsigned int tag_max);
        virtual ~ParticleSelectorTag() {}
    };

//! ParticleSelector to select particles based on the type rule
class ParticleSelectorType : public ParticleSelectorRule<TypeRule>
    {
    public:
        //! Constructs the selector
        ParticleSelectorType(boost::shared_ptr<SystemDefinition> sysdef, unsigned int typ_min, unsigned int typ_max);
        virtual ~ParticleSelectorType() {}
    };

//! Select particles based on their rigid body
class ParticleSelectorRigid : public ParticleSelectorRule<RigidRule>
    {
    public:
        //! Constructs the selector
        ParticleSelectorRigid(boost::shared_ptr<SystemDefinition> sysdef, bool rigid);
        virtual ~ParticleSelectorRigid() {}
    };

//! Select particles based on a list of global tags to include
class ParticleSelectorGlobalTagList : public ParticleSelector
    {
    public:
        //! constructs this particle selector
        /*! \param sysdef System definition to use for selecting particles
         * \param global_tag_list List of global tags to include in the group
         */
        ParticleSelectorGlobalTagList(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list);
        virtual ~ParticleSelectorGlobalTagList() {}

        //! Get the list of selected tags
        /*! \param member_tags GPU array to store the member tags in
         * \return number of local particles included
         * \pre  member_tags must be allocated and of sufficient size to accomodate
         *       all local members of the group (i.e.
         *       the current maximum number of particles returned by ParticleData::getMaxN() )
        */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags);

    protected:
        GPUArray<unsigned int> m_global_member_tags;    //!< array of global member tags

        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience
    };

//! ParticleSelector that represents performs a set operation on two ParticleSelectors (abstract base class)
class ParticleSelectorSetOperation : public ParticleSelector
    {
    public:
        //! constructs this particle selector
        /*! \param sysdef System definition used for storing internal tag lists
            \param a first selector to perform the set operation on
            \param b second selector to perform the set operation on
         */
        ParticleSelectorSetOperation(boost::shared_ptr<SystemDefinition> sysdef,
                              boost::shared_ptr<ParticleSelector> a,
                              boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorSetOperation() {}

        //! Get the list of selected tags
        /*! \param member_tags GPU array to store the member tags in
         * \returns number of local particles included
         * \pre  member_tags must be allocated and of sufficient size to accomodate
         *       all local members of the group (i.e.
         *       the current maximum number of particles returned by ParticleData::getMaxN() )
        */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags);

    protected:
        //! Perform the set operation between both selectors
        /*! \param member_tags_a GPU array of local members returned by the first selector
         *  \param member_tags_b GPU array of local members returned by the second selector
         *  \param num_members_a number of local members returned by the first selector
         *  \param num_members_b number of local members returned by the second selector
         *  \param member_tags GPU array for storing members of the combined selection
         *  \returns number of members in the combined selection
         */
        virtual unsigned operation(const GPUArray<unsigned int>& member_tags_a,
                                      const GPUArray<unsigned int>& member_tags_b,
                                      const unsigned int num_members_a,
                                      const unsigned int num_members_b,
                                      const GPUArray<unsigned int>& member_tags) = 0;
    private:
        GPUArray<unsigned int> m_member_tags_a;            //!< Local members of first particle selector
        GPUArray<unsigned int> m_member_tags_b;            //!< Local members of second particle selector

        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience

        boost::shared_ptr<ParticleSelector> m_selector_a;  //!< first argument
        boost::shared_ptr<ParticleSelector> m_selector_b;  //!< second argument
    };

//! ParticleSelector that represents a union of two ParticleSelectors
class ParticleSelectorUnion : public ParticleSelectorSetOperation
    {
    public:
        //! constructs this particle selector
        /*! \param sysdef System definition used for storage of internal tags
            \param a first selector to include in the union
            \param b second selector to include in the union
         */
        ParticleSelectorUnion(boost::shared_ptr<SystemDefinition> sysdef,
                              boost::shared_ptr<ParticleSelector> a,
                              boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorUnion() {}

    protected:
        //! Perform the union between both selectors
        /*! \param member_tags_a GPU array of local members returned by the first selector
         *  \param member_tags_b GPU array of local members returned by the second selector
         *  \param num_members_a number of local members returned by the first selector
         *  \param num_members_b number of local members returned by the second selector
         *  \param member_tags GPU array for storing members of the combined selection
         *  \returns number of members in the combined selection
         */
        virtual unsigned operation(const GPUArray<unsigned int>& member_tags_a,
                                   const GPUArray<unsigned int>& member_tags_b,
                                   const unsigned int num_members_a,
                                   const unsigned int num_members_b,
                                   const GPUArray<unsigned int>& member_tags);
    };

//! ParticleSelector that represents an intersection of two ParticleSelectors
class ParticleSelectorIntersection : public ParticleSelectorSetOperation
    {
    public:
        //! constructs this particle selector
        /*! \param sysdef System definition to use for storing the internal tags
            \param a first selector to perform the intersection on
            \param b second selector to perform the intersection on
         */
        ParticleSelectorIntersection(boost::shared_ptr<SystemDefinition> sysdef,
                                     boost::shared_ptr<ParticleSelector> a,
                                     boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorIntersection() {}

    protected:
        //! Make the intersection between the two selectors
        /*! \param member_tags_a GPU array of local members returned by the first selector
         *  \param member_tags_b GPU array of local members returned by the second selector
         *  \param num_members_a number of local members returned by the first selector
         *  \param num_members_b number of local members returned by the second selector
         *  \param member_tags GPU array for storing members of the combined selection
         *  \returns number of members in the combined selection
         */
        virtual unsigned operation(const GPUArray<unsigned int>& member_tags_a,
                                   const GPUArray<unsigned int>& member_tags_b,
                                   const unsigned int num_members_a,
                                   const unsigned int num_members_b,
                                   const GPUArray<unsigned int>& member_tags);
    };

//! ParticleSelector that represents the difference of two ParticleSelectors
class ParticleSelectorDifference : public ParticleSelectorSetOperation
    {
    public:
        //! constructs this particle selector
        /*! \param sysdef System definition to use for storing the tags
            \param a first selector to perform the intersection on
            \param b second selector to perform the intersection on
         */
        ParticleSelectorDifference(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleSelector> a,
                                   boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorDifference() {}

    protected:
        //! Make the difference between the two selectors

        /*! \param member_tags_a GPU array of local members returned by the first selector
         *  \param member_tags_b GPU array of local members returned by the second selector
         *  \param num_members_a number of local members returned by the first selector
         *  \param num_members_b number of local members returned by the second selector
         *  \param member_tags GPU array for storing members of the combined selection
         *  \returns number of members in the combined selection
         */
        virtual unsigned operation(const GPUArray<unsigned int>& member_tags_a,
                                   const GPUArray<unsigned int>& member_tags_b,
                                   const unsigned int num_members_a,
                                   const unsigned int num_members_b,
                                   const GPUArray<unsigned int>& member_tags);
    };

//! The trivial particle selector that represents the empty set
class ParticleSelectorEmptySet : public ParticleSelector
    {
    public:
        //! Constructs the particle selector
        ParticleSelectorEmptySet() {}
        virtual ~ParticleSelectorEmptySet() {}

        //! Get the number of selected local particles
        /*! \param member_tags GPU array to store member tags in
         * \returns zero
         */
        virtual unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags)
            {
            return 0;
            }
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

    Group membership is determined once at the instantiation of the group. This means, that in terms of global
    particle tags the membership does not change over the course of a simulation. Internally, however, a ParticleGroup
    maintains a current list of local member. This list may change as particles are inserted in or deleted from
    the simulation domain. Truly dynamic groups, if they are needed, may require a drastically different design
    to allow for efficient access and update.

    In many use-cases, ParticleGroup may be accessed many times within inner loops. Thus, it must not aquire any
    ParticleData arrays within most of the get() calls as the caller must be allowed to leave their ParticleData 
    aquired. Thus, all get() methods must return values from internal cached variables only. Those methods that
    absolutely require the particle data be released before they are called will be documented as such.

    <b>Data Structures and Implementation</b>

    FIXME: Currently, the sorted tag order of local group members is no longer guaranteed. In fact, would (local) sorted
    order make sense at all? We might need to rethink how and when getMemberTag() is used in a simulation. If it is only
    used outside a simulation in order to get a list group members in sorted order, we might require for its use
    that all global particles be present in the local ParticleData. We need to find a way to sort the group members, then.

    The initial and fundamental data structure in the group is an array of all particle tags in the group,
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
        ParticleGroup();

        //! Constructs a particle group of all particles that meet the given selection
        ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleSelector> selector);

        //! Constructs a particle group of particles with global tags given by a list
        ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list);

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
            return m_num_members;
            }
            
        //! Get a member from the group
        /*! \param i Index from 0 to getNumMembers()-1 of the group member to get
            \returns Tag of the member at index \a i

            FIXME: This is currently broken, since not all global group members may be present on the local
                   processor. We need to come up with a solution e.g. in which this function is only
                   used on processor with rank 0, only if it has all the data available. On this processor,
                   the tags should be in sorted order...
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
            assert(j < getNumMembers());
            ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::read);
            return h_handle.data[j];
            }

        //! Test if a particle index is a member of the group
        /*! \param idx Index of the particle to query (from 0 to the number of partilces in ParticleData -1)
            \returns true if the particle with index \a idx is in the group
        */
        bool isMember(unsigned int idx) const
            {
            if (m_is_empty) return false;
            ArrayHandle<unsigned char> h_is_member(m_is_member, access_location::host, access_mode::read);
            return (h_is_member.data[idx] == 1);
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
        GPUArray<unsigned char> m_is_member;            //!< One byte per particle, =1 if index is a member of the group, =0 otherwise
        boost::shared_ptr<ParticleSelector> m_selector; //!< The particle selector associated with this particle group
        GPUArray<unsigned int> m_member_idx;            //!< List of all particle indices in the group
        boost::signals::connection m_sort_connection;   //!< Connection to the ParticleData sort signal
        boost::signals::connection m_particle_num_change_connection; //!< Connection to the ParticleData particle number change signal
        boost::signals::connection m_max_particle_num_change_connection; //!< Connection to the maximum number change signal
        GPUArray<unsigned int> m_member_tags;           //!< Lists the tags of local particle members
        unsigned int m_num_members;                     //!< Number of local members of the group

        bool m_is_empty;                                //!< True if this is an empty group

        //! Helper function to resize array of member tags
        void reallocate();

        //! Helper function to rebuild the index lists afer the particles have been sorted
        void rebuildIndexList();

#ifdef ENABLE_CUDA
        //! Helper function to rebuild the index lists afer the particles have been sorted
        void rebuildIndexListGPU();
#endif

        //! Helper function called when rebuilding the tag list is necessary
        void rebuildTagList();
    };


//! Exports the ParticleGroup class to python
void export_ParticleGroup();

#endif

