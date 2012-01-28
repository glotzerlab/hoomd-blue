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
    which internally store clones of the ParticleSelector classes of the ParticleGroups they take as an argument.
    
    <b>Implementation details</b>
    As the ParticleSelector is specified via a virtual class the group definition can be expanded to include any
    concievable selection criteria.

    So that an infinite range of selection criteria can be applied (i.e. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a reference to the SystemDefinition on construction, along with
    any parameters to specify the selection criteria.

    The ParticleSelector base class only provides an interface. An important implementation is given by the
    rule-based particle ParticleSelectorRule<>, it is a template class that takes a rule as a parameter. This rule
    can be efficiently evaluated on the CPU or on the GPU to construct the tag list.
    
    ParticleSelector provides a purely virtual internal method rebuildTagsList()
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
        /*! \return the tag list
        */
        virtual std::vector<unsigned int>& getMemberTags();

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        virtual ParticleSelector* clone() = 0;

        friend class ParticleGroup;
        friend class ParticleSelectorUnion;
        friend class ParticleSelectorIntersection;
        friend class ParticleSelectorDifference;

    protected:
        std::vector<unsigned int> m_member_tags;        //!< List of global tags that are owned by this processor and that are part of the group

        //! Helper function to rebuild the tags list
        virtual void rebuildTagsList() = 0;
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

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        ParticleSelector* clone()
            {
            return new ParticleSelectorRule<T>(*this);
            }

    protected:
        //! Internal function to rebuild the tags list
        virtual void rebuildTagsList();

        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience
    private:
        typename T::param_type m_params; //!< rule parameters
    };

//! Constructor
template< class T >
ParticleSelectorRule<T>::ParticleSelectorRule(boost::shared_ptr<SystemDefinition> sysdef)
    : ParticleSelector(), m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData())
    {
    }

//! Loop over particles and determine those that match the selection criterium
template< class T >
void ParticleSelectorRule<T>::rebuildTagsList()
    {
    // reset tags list
    m_member_tags.clear();

    // instantiate the selection rule
    T rule(m_params);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        // add the tag to the list if it matches the selection
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        unsigned int body = h_body.data[idx];
        unsigned int global_tag = h_global_tag.data[idx];

        if (rule.isSelected(global_tag, body, type))
            m_member_tags.push_back(global_tag);
        }
    }

//! Set the parameters for the rule
template< class T >
void ParticleSelectorRule<T>::setParams(typename T::param_type params)
    {
    m_params = params;
    }

//! Rule to select particles based on their global tag
class GlobalTagRule
    {
    public:
        typedef uint2 param_type; //!< parameter type for storing minimum and maximum tag

        //! Constructor
        //! \param params parameters for this rule
        GlobalTagRule(param_type params);

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type);

    private:
        unsigned int _tag_min;     //! Minimum global tag to select
        unsigned int _tag_max;     //! Maximum global tag to select
    };

//! Rule to select particles based on their type
class TypeRule
    {
    public:
        typedef uint2 param_type; //!< parameter type for storing minimum and maximum particle type

        //! Constructor
        //! \param params parameters for this rule
        TypeRule(param_type params);

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type);

    private:
        unsigned int _type_min;     //! Minimum particle tag to select
        unsigned int _type_max;     //! Maximum particle tag to select
    };

//! Rule to select particles that are in rigid bodies
class RigidRule
    {
    public:
        typedef bool param_type;   //!< parameter type

        //! Constructor
        //! \param param parameter for this rule
        RigidRule(param_type param);

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type);
    private:
        bool _rigid; //!<true selects particles that are in rigid bodies, false selects particles that are not part of a body
    };

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
        ParticleSelectorGlobalTagList(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list);
        virtual ~ParticleSelectorGlobalTagList() {}

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        ParticleSelector *clone()
            {
            return new ParticleSelectorGlobalTagList(*this);
            }

    protected:

        //! Internal function to rebuild the tags list
        virtual void rebuildTagsList();

        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience

    private:
        const std::vector<unsigned int> m_global_member_tags; //!< List of global member tags
    };

//! ParticleSelector that represents a union of two ParticleSelectors
class ParticleSelectorUnion : public ParticleSelector
    {
    public:
        //! constructs this particle selector
        /*! \param a first selector to include in the union
            \param b second selector to include in the union
         */
        ParticleSelectorUnion(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorUnion() {}

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        ParticleSelector *clone()
            {
            return new ParticleSelectorUnion(*this);
            }

    protected:

        //! Internal function to rebuild the tags list
        virtual void rebuildTagsList();

    private:
        boost::shared_ptr<ParticleSelector> m_selector_a;  //!< first argument
        boost::shared_ptr<ParticleSelector> m_selector_b;  //!< second argument
    };

//! ParticleSelector that represents an intersection of two ParticleSelectors
class ParticleSelectorIntersection : public ParticleSelector
    {
    public:
        //! constructs this particle selector
        /*! \param a first selector to perform the intersection on
            \param b second selector to perform the intersection on
         */
        ParticleSelectorIntersection(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorIntersection() {}

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        ParticleSelector *clone()
            {
            return new ParticleSelectorIntersection(*this);
            }

    protected:

        //! Internal function to rebuild the tags list
        virtual void rebuildTagsList();

    private:
        boost::shared_ptr<ParticleSelector> m_selector_a;  //!< first argument
        boost::shared_ptr<ParticleSelector> m_selector_b;  //!< second argument
    };

//! ParticleSelector that represents the difference of two ParticleSelectors
class ParticleSelectorDifference : public ParticleSelector
    {
    public:
        //! constructs this particle selector
        /*! \param a first selector to perform the intersection on
            \param b second selector to perform the intersection on
         */
        ParticleSelectorDifference(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b);
        virtual ~ParticleSelectorDifference() {}

        //! Clone this ParticleSelector
        /*! \return a copy of the ParticleSelector
        */
        ParticleSelector *clone()
            {
            return new ParticleSelectorDifference(*this);
            }

    protected:

        //! Internal function to rebuild the tags list
        virtual void rebuildTagsList();

    private:
        boost::shared_ptr<ParticleSelector> m_selector_a;  //!< first argument
        boost::shared_ptr<ParticleSelector> m_selector_b;  //!< second argument
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
            return (unsigned int)m_member_tags.size();
            }
            
        //! Get a member from the group
        /*! \param i Index from 0 to getNumMembers()-1 of the group member to get
            \returns Tag of the member at index \a i
        */
        unsigned int getMemberTag(unsigned int i) const
            {
            assert(i < getNumMembers());
            return m_member_tags[i];
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
            return m_is_member[idx];
            }
        
        //! Direct access to the index list
        /*! \returns A GPUArray for directly accessing the index list, intended for use in using groups on the GPU
            \note The caller \b must \b not write to or change the array.
        */
        const GPUArray<unsigned int>& getIndexArray() const
            {
            return m_member_idx;
            }

        //! Update the group members
        /*! \param timestep current timestep of the simulation
            The default implementation is empty, as the members of a ParticleGroup are not supposed to change during the
            simulation. In derived classes this method may be reimplemented to make the group dynamic.
         */
         void update(unsigned int timestep)
             {
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
        boost::dynamic_bitset<> m_is_member;            //!< One bit per particle, true if index is a member of the group
        boost::shared_ptr<ParticleSelector> m_selector; //!< The particle selector associated with this particle group
        GPUArray<unsigned int> m_member_idx;            //!< List of all particle indices in the group
        boost::signals::connection m_sort_connection;   //!< Connection to the ParticleData sort signal
        boost::signals::connection m_particle_num_change_connection; //!< Connection to the ParticleData particle number change signal
        std::vector<unsigned int>& m_member_tags;       //!< Lists the tags of the paritcle members

        //! Helper function to rebuild the index lists afer the particles have been sorted
        void rebuildIndexList();

        //! Helper function called when rebuilding the tag list is necessary
        void rebuildTagList();
    };


//! Exports the ParticleGroup class to python
void export_ParticleGroup();

#endif

