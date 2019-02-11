// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ParticleGroup.h
    \brief Declares the ParticleGroup and related classes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SystemDefinition.h"

#include <string>
#include <memory>
#include <vector>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "GlobalArray.h"

#ifdef ENABLE_CUDA
#include "GPUPartition.cuh"
#endif

#ifndef __PARTICLE_GROUP_H__
#define __PARTICLE_GROUP_H__

//! Utility class to select particles based on given conditions
/*! \b Overview

    In order to flexibly specify the particles that belong to a given ParticleGroup, it will simple take a
    ParticleSelector as a parameter in its constructor. The selector will provide a true/false membership test that will
    be applied to each particle tag, selecting those that belong in the group. As it is specified via a virtual class,
    the group definition can be expanded to include any conceivable selection criteria.

    <b>Implementation details</b>
    So that an infinite range of selection criteria can be applied (i.e. particles with mass > 2.0, or all particles
    bonded to particle j, ...) the selector will get a reference to the SystemDefinition on construction, along with
    any parameters to specify the selection criteria. Then, a simple getSelectedTags() call will return
    a list of particle tags meeting the criteria.

    In parallel simulations, getSelectedTags() should return only local tags.

    The base class getSelectedTags() method will simply return an empty list.
    selection semantics.
*/
class PYBIND11_EXPORT ParticleSelector
    {
    public:
        //! constructs a ParticleSelector
        ParticleSelector(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelector() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;

    protected:
        std::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition assigned to this selector
        std::shared_ptr<ParticleData> m_pdata;        //!< The particle data from m_sysdef, stored as a convenience
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
    };

//! Select all particles
class PYBIND11_EXPORT ParticleSelectorAll : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorAll(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelectorAll() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    };


//! Select particles based on their tag
class PYBIND11_EXPORT ParticleSelectorTag : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorTag(std::shared_ptr<SystemDefinition> sysdef, unsigned int tag_min, unsigned int tag_max);
        virtual ~ParticleSelectorTag() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        unsigned int m_tag_min;     //!< Minimum tag to select
        unsigned int m_tag_max;     //!< Maximum tag to select (inclusive)
    };

//! Select particles based on their type
class PYBIND11_EXPORT ParticleSelectorType : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorType(std::shared_ptr<SystemDefinition> sysdef, unsigned int typ_min, unsigned int typ_max);
        virtual ~ParticleSelectorType() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        unsigned int m_typ_min;     //!< Minimum type to select
        unsigned int m_typ_max;     //!< Maximum type to select (inclusive)
    };

//! Select particles in the space defined by a cuboid
class PYBIND11_EXPORT ParticleSelectorCuboid : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorCuboid(std::shared_ptr<SystemDefinition> sysdef, Scalar3 min, Scalar3 max);
        virtual ~ParticleSelectorCuboid() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        Scalar3 m_min;     //!< Minimum type to select (inclusive)
        Scalar3 m_max;     //!< Maximum type to select (exclusive)
    };

//! Select particles based on their rigid body
class PYBIND11_EXPORT ParticleSelectorRigid : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorRigid(std::shared_ptr<SystemDefinition> sysdef, bool rigid);
        virtual ~ParticleSelectorRigid() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        bool m_rigid;   //!< true if we should select particles in rigid bodies, false if we should select non-rigid particles
    };

//! Select particles based on their body
class PYBIND11_EXPORT ParticleSelectorBody : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorBody(std::shared_ptr<SystemDefinition> sysdef, bool body);
        virtual ~ParticleSelectorBody() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        bool m_body;   //!< true if we should select particles in a body, false if we should select non-body particles
    };

//! Select particles based on their floppy body
class PYBIND11_EXPORT ParticleSelectorFloppy : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorFloppy(std::shared_ptr<SystemDefinition> sysdef, bool molecule);
        virtual ~ParticleSelectorFloppy() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
    protected:
        bool m_floppy;   //!< true if we should select particles in floppy bodies, false if we should select non-floppy particles
    };

class PYBIND11_EXPORT ParticleSelectorRigidCenter : public ParticleSelector
    {
    public:
        //! Constructs the selector
        ParticleSelectorRigidCenter(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~ParticleSelectorRigidCenter() {}

        //! Test if a particle meets the selection criteria
        virtual std::vector<unsigned int> getSelectedTags() const;
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

    In many use-cases, ParticleGroup may be accessed many times within inner loops. Thus, it must not acquire any
    ParticleData arrays within most of the get() calls as the caller must be allowed to leave their ParticleData
    acquired. Thus, all get() methods must return values from internal cached variables only. Those methods that
    absolutely require the particle data be released before they are called will be documented as such.

    <b>Data Structures and Implementation</b>

    The initial and fundamental data structure in the group is a vector listing all of the particle tags in the group,
    in a sorted tag order. This list can be accessed directly via getMemberTag() to meet the 2nd use case listed above.
    In order to iterate through all particles in the group in a cache-efficient manner, an auxiliary list is stored
    that lists all particle <i>indices</i> that belong to the group. This list must be updated on every particle sort.
    Thirdly, a dynamic bitset is used to store one bit per particle for efficient O(1) tests if a given particle is in
    the group.

    Finally, the common use case on the GPU using groups will include running one thread per particle in the group.
    For that it needs a list of indices of all the particles in the group. To facilitates this, the list of indices
    in the group will be stored in a GPUArray.

    \ingroup data_structs
*/
class PYBIND11_EXPORT ParticleGroup
    {
    public:
        //! \name Initialization methods
        // @{

        //! Constructs an empty particle group
        ParticleGroup() : m_num_local_members(0) {};

        //! Constructs a particle group of all particles that meet the given selection
        ParticleGroup(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleSelector> selector,
            bool update_tags = true);

        //! Constructs a particle group given a list of tags
        ParticleGroup(std::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& member_tags);

        //! Destructor
        ~ParticleGroup();

        //! Updates the members tags of a particle group according to a selection
        void updateMemberTags(bool force_update) const;

        // @}
        //! \name Accessor methods
        // @{

        //! Get the number of members in the group
        /*! \returns The number of particles that belong to this group
        */
        unsigned int getNumMembersGlobal() const
            {
            checkRebuild();

            return (unsigned int)m_member_tags.getNumElements();
            }

        //! Get the number of members that are present on the local processor
        /*! \returns The number of particles on the local processor that belong to this group
        */
        unsigned int getNumMembers() const
            {
            checkRebuild();

            return m_num_local_members;
            }

        //! Get a member from the group
        /*! \param i Index from 0 to getNumMembersGlobal()-1 of the group member to get
            \returns Tag of the member at index \a i
        */
        unsigned int getMemberTag(unsigned int i) const
            {
            checkRebuild();

            assert(i < getNumMembersGlobal());
            ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::read);
            return h_member_tags.data[i];
            }

        //! Get a member index from the group
        /*! \param j Value from 0 to getNumMembers()-1 of the group member to get
            \returns Index of the member at position \a j
            \note getMemberTag(j) \b does \b NOT get the tag of the particle with index getMemberIndex(j). These two
                  lists are stored in different orders. Access the ParticleData to convert between tags and indices.
            \note This method CAN access the particle data tag array if the index is rebuilt.
                  Hence, the tag array may not be accessed in the same scope in which this method is called.
        */
        unsigned int getMemberIndex(unsigned int j) const
            {
            checkRebuild();

            assert(j < getNumMembers());
            ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::read);
            unsigned int idx = h_handle.data[j];
            assert(idx < m_pdata->getN());
            return idx;
            }

        //! Test if a particle index is a member of the group
        /*! \param idx Index of the particle to query (from 0 to the number of particles in ParticleData -1)
            \returns true if the particle with index \a idx is in the group
            \note This method CAN access the particle data tag array if the index is rebuilt.
                  Hence, the tag array may not be accessed in the same scope in which this method is called.
        */
        bool isMember(unsigned int idx) const
            {
            checkRebuild();

            ArrayHandle<unsigned int> h_handle(m_is_member, access_location::host, access_mode::read);
            return h_handle.data[idx] == 1;
            }

        //! Direct access to the index list
        /*! \returns A GPUArray for directly accessing the index list, intended for use in using groups on the GPU
            \note The caller \b must \b not write to or change the array.

            \note This method CAN access the particle data tag array if the index is rebuilt.
                  Hence, the tag array may not be accessed in the same scope in which this method is called.
        */
        const GlobalArray<unsigned int>& getIndexArray() const
            {
            checkRebuild();

            return m_member_idx;
            }

        #ifdef ENABLE_CUDA
        //! Return the load balancing GPU partition
        const GPUPartition& getGPUPartition() const
            {
            checkRebuild();

            return m_gpu_partition;
            }
        #endif

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
        static std::shared_ptr<ParticleGroup> groupUnion(std::shared_ptr<ParticleGroup> a,
                                                           std::shared_ptr<ParticleGroup> b);
        //! Make a new particle group from an intersection
        static std::shared_ptr<ParticleGroup> groupIntersection(std::shared_ptr<ParticleGroup> a,
                                                                  std::shared_ptr<ParticleGroup> b);
        //! Make a new particle group from an difference
        static std::shared_ptr<ParticleGroup> groupDifference(std::shared_ptr<ParticleGroup> a,
                                                                std::shared_ptr<ParticleGroup> b);

        // @}

    private:
        std::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition this group is associated with
        std::shared_ptr<ParticleData> m_pdata;        //!< The particle data this group is associated with
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration

        // NOTE a design with so many mutable members is broken, we should refactor const correctness
        // in ParticleGroup in the future by using resize methods on the arrays
        mutable GlobalArray<unsigned int> m_is_member;    //!< One byte per particle, == 1 if index is a local member of the group
        mutable GlobalArray<unsigned int> m_member_idx;    //!< List of all particle indices in the group
        mutable GlobalArray<unsigned int> m_member_tags;   //!< Lists the tags of the particle members
        mutable unsigned int m_num_local_members;       //!< Number of members on the local processor
        mutable bool m_particles_sorted;                //!< True if particle have been sorted since last rebuild
        mutable bool m_reallocated;                     //!< True if particle data arrays have been reallocated
        mutable bool m_global_ptl_num_change;           //!< True if the global particle number changed

        mutable GlobalArray<unsigned int> m_is_member_tag;  //!< One byte per particle, == 1 if tag is a member of the group
        std::shared_ptr<ParticleSelector> m_selector; //!< The associated particle selector

        bool m_update_tags;                             //!< True if tags should be updated when global number of particles changes
        mutable bool m_warning_printed;                         //!< True if warning about static groups has been printed

        #ifdef ENABLE_CUDA
        mutable GPUPartition m_gpu_partition;           //!< A handy struct to store load balancing info for this group's local members
        #endif

        //! Helper function to resize array of member tags
        void reallocate() const;

        //! Helper function to rebuild the index lists after the particles have been sorted
        void rebuildIndexList() const;

        //! Helper function to rebuild internal arrays
        void checkRebuild() const
            {
            // carry out rebuild in correct order
            bool update_gpu_advice = false;
            if (m_global_ptl_num_change)
                {
                updateMemberTags(false);
                m_global_ptl_num_change = false;
                }
            if (m_reallocated)
                {
                reallocate();
                m_reallocated = false;
                update_gpu_advice = true;
                }
             if (m_particles_sorted)
                {
                rebuildIndexList();
                m_particles_sorted = false;
                }
            if (update_gpu_advice)
                {
                updateGPUAdvice();
                }
            }

        //! Helper function to be called when the particle data arrays are reallocated
        void slotReallocate()
            {
            m_reallocated = true;
            }

        //! Helper function to be called when the particles are resorted
        void slotParticleSort()
            {
            m_particles_sorted = true;
            }

        //! Update the GPU memory advice
        void updateGPUAdvice() const;

        //! Helper function to be called when particles are added/removed
        void slotGlobalParticleNumChange()
            {
            m_global_ptl_num_change = true;
            }

        //! Helper function to build the 1:1 hash for tag membership
        void buildTagHash() const;

#ifdef ENABLE_CUDA
        //! Helper function to rebuild the index lists after the particles have been sorted
        void rebuildIndexListGPU() const;
#endif

    };

//! Exports the ParticleGroup class to python
void export_ParticleGroup(pybind11::module& m);

#endif
