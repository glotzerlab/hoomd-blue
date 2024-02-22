// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ParticleGroup.h
    \brief Declares the ParticleGroup and related classes
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "SystemDefinition.h"
#include "filter/ParticleFilter.h"

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "GlobalArray.h"

#ifdef ENABLE_HIP
#include "GPUPartition.cuh"
#endif

#ifndef __PARTICLE_GROUP_H__
#define __PARTICLE_GROUP_H__

namespace hoomd
    {
//! Describes a group of particles
/*! \b Overview

    Some computations in HOOMD need to only be performed on certain groups of particles.
   ParticleGroup facilitates that by providing a flexible interface for choosing these groups that
   can be used by any other class in HOOMD.

    The following common use-cases are expected and the design is tuned to make these optimal.
     - Iterate through all particles indices in the group, and the order of iteration doesn't
   matter, except for performance.
     - Iterate through all particle tags in the group, in a well-defined order that does not change
   (namely, a sorted tag order is required)
     - O(1) test if a particular particle index is in the group

    Membership in the group is determined through a generic ParticleFilter class. See its
   documentation for details.

    Group membership is determined once at the instantiation of the group. Thus ParticleGroup only
   supports static groups where membership does not change over the course of a simulation. Dynamic
   groups, if they are needed, may require a drastically different design to allow for efficient
   access.

    In many use-cases, ParticleGroup may be accessed many times within inner loops. Thus, it must
   not acquire any ParticleData arrays within most of the get() calls as the caller must be allowed
   to leave their ParticleData acquired. Thus, all get() methods must return values from internal
   cached variables only. Those methods that absolutely require the particle data be released before
   they are called will be documented as such.

    <b>Data Structures and Implementation</b>

    The initial and fundamental data structure in the group is a vector listing all of the particle
   tags in the group, in a sorted tag order. This list can be accessed directly via getMemberTag()
   to meet the 2nd use case listed above. In order to iterate through all particles in the group in
   a cache-efficient manner, an auxiliary list is stored that lists all particle <i>indices</i> that
   belong to the group. This list must be updated on every particle sort. Thirdly, a dynamic bitset
   is used to store one bit per particle for efficient O(1) tests if a given particle is in the
   group.

    Finally, the common use case on the GPU using groups will include running one thread per
   particle in the group. For that it needs a list of indices of all the particles in the group. To
   facilitates this, the list of indices in the group will be stored in a GPUArray.

    \ingroup data_structs
*/
class PYBIND11_EXPORT ParticleGroup
    {
    public:
    //! \name Initialization methods
    // @{

    //! Constructs a particle group of all particles that meet the given selection
    ParticleGroup(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<ParticleFilter> selector,
                  bool update_tags = true);

    //! Constructs a particle group given a list of tags
    ParticleGroup(std::shared_ptr<SystemDefinition> sysdef,
                  const std::vector<unsigned int>& member_tags);

    //! Destructor
    ~ParticleGroup();

    std::shared_ptr<ParticleFilter> getFilter()
        {
        return m_selector;
        }

    //! Updates the members tags of a particle group according to a selection
    void updateMemberTags(bool force_update);

    // @}
    //! \name Accessor methods
    // @{

    //! Get the number of members in the group
    /*! \returns The number of particles that belong to this group
     */
    unsigned int getNumMembersGlobal()
        {
        checkRebuild();

        return (unsigned int)m_member_tags.getNumElements();
        }

    //! Get the number of members that are present on the local processor
    /*! \returns The number of particles on the local processor that belong to this group
     */
    unsigned int getNumMembers()
        {
        checkRebuild();

        return m_num_local_members;
        }

    //! Get a member from the group
    /*! \param i Index from 0 to getNumMembersGlobal()-1 of the group member to get
        \returns Tag of the member at index \a i
    */
    unsigned int getMemberTag(unsigned int i)
        {
        checkRebuild();

        assert(i < getNumMembersGlobal());
        ArrayHandle<unsigned int> h_member_tags(m_member_tags,
                                                access_location::host,
                                                access_mode::read);
        return h_member_tags.data[i];
        }

    //! Get a member index from the group
    /*! \param j Value from 0 to getNumMembers()-1 of the group member to get
        \returns Index of the member at position \a j
        \note getMemberTag(j) \b does \b NOT get the tag of the particle with index
       getMemberIndex(j). These two lists are stored in different orders. Access the ParticleData to
       convert between tags and indices. \note This method CAN access the particle data tag array if
       the index is rebuilt. Hence, the tag array may not be accessed in the same scope in which
       this method is called.
    */
    unsigned int getMemberIndex(unsigned int j)
        {
        checkRebuild();

        assert(j < getNumMembers());
        ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::read);
        unsigned int idx = h_handle.data[j];
        assert(idx < m_pdata->getN());
        return idx;
        }

    //! Test if a particle index is a member of the group
    /*! \param idx Index of the particle to query (from 0 to the number of particles in ParticleData
       -1) \returns true if the particle with index \a idx is in the group \note This method CAN
       access the particle data tag array if the index is rebuilt. Hence, the tag array may not be
       accessed in the same scope in which this method is called.
    */
    bool isMember(unsigned int idx)
        {
        checkRebuild();

        ArrayHandle<unsigned int> h_handle(m_is_member, access_location::host, access_mode::read);
        return h_handle.data && h_handle.data[idx] == 1;
        }

    //! Direct access to the index list
    /*! \returns A GPUArray for directly accessing the index list, intended for use in using groups
       on the GPU \note The caller \b must \b not write to or change the array.

        \note This method CAN access the particle data tag array if the index is rebuilt.
              Hence, the tag array may not be accessed in the same scope in which this method is
       called.
    */
    const GlobalArray<unsigned int>& getIndexArray()
        {
        checkRebuild();

        return m_member_idx;
        }

#ifdef ENABLE_HIP
    //! Return the load balancing GPU partition
    const GPUPartition& getGPUPartition()
        {
        checkRebuild();

        return m_gpu_partition;
        }
#endif

    // @}
    //! \name Analysis methods
    // @{

    //! Compute the total mass of the group
    Scalar getTotalMass();
    //! Compute the center of mass of the group
    Scalar3 getCenterOfMass();

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

    /// Set the number of degrees of freedom
    void setTranslationalDOF(Scalar dof)
        {
        m_translational_dof = dof;
        }

    /// Get the number translational degrees of freedom
    Scalar getTranslationalDOF()
        {
        return m_translational_dof;
        }

    /// Set the number of degrees of freedom
    void setRotationalDOF(Scalar dof)
        {
        m_rotational_dof = dof;
        }

    /// Get the number of degrees of freedom
    Scalar getRotationalDOF()
        {
        return m_rotational_dof;
        }

    /// Get a NumPy array of the the local member tags.
    /** This is necessary to enable testing in Python the updating of ParticleGroup instances.
     */
    pybind11::array_t<unsigned int> getMemberTags() const
        {
        const ArrayHandle<unsigned int> h_member_tags(m_member_tags,
                                                      access_location::host,
                                                      access_mode::read);
        return pybind11::array_t<unsigned int, pybind11::array::c_style>(
            static_cast<size_t>(m_member_tags.getNumElements()),
            h_member_tags.data);
        }

    /** Get the number of particles present in both groups.

        @param other Second group
    */
    unsigned int intersectionSize(std::shared_ptr<ParticleGroup> other);

    /** Thermalize the particle momenta

        Draw Gaussian random momenta for all particles in the group.

        @param kT Thermal energy
        @param timestep The current simulation timestep
    */
    void thermalizeParticleMomenta(Scalar kT, uint64_t timestep);

    /// Get the number of central and free particles (global)
    unsigned int getNCentralAndFreeGlobal()
        {
        return m_n_central_and_free_global;
        }

    private:
    std::shared_ptr<SystemDefinition>
        m_sysdef;                          //!< The system definition this group is associated with
    std::shared_ptr<ParticleData> m_pdata; //!< The particle data this group is associated with
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration

    // NOTE a design with so many mutable members is broken, we should refactor const correctness
    // in ParticleGroup in the future by using resize methods on the arrays
    mutable GlobalArray<unsigned int>
        m_is_member; //!< One byte per particle, == 1 if index is a local member of the group
    mutable GlobalArray<unsigned int> m_member_idx;  //!< List of all particle indices in the group
    mutable GlobalArray<unsigned int> m_member_tags; //!< Lists the tags of the particle members
    mutable unsigned int m_num_local_members;        //!< Number of members on the local processor
    mutable bool m_particles_sorted;      //!< True if particle have been sorted since last rebuild
    mutable bool m_reallocated;           //!< True if particle data arrays have been reallocated
    mutable bool m_global_ptl_num_change; //!< True if the global particle number changed

    mutable GlobalArray<unsigned int>
        m_is_member_tag; //!< One byte per particle, == 1 if tag is a member of the group
    std::shared_ptr<ParticleFilter> m_selector; //!< The associated particle selector

    bool m_update_tags; //!< True if tags should be updated when global number of particles changes
    mutable bool m_warning_printed; //!< True if warning about static groups has been printed

#ifdef ENABLE_HIP
    mutable GPUPartition m_gpu_partition; //!< A handy struct to store load balancing info for this
                                          //!< group's local members
#endif

    /// Number of translational degrees of freedom in the group
    Scalar m_translational_dof = 0;

    /// Number of rotational degrees of freedom in the group
    Scalar m_rotational_dof = 0;

    /// Number of central and free particles in the group (global)
    unsigned int m_n_central_and_free_global = 0;

    //! Helper function to resize array of member tags
    void reallocate();

    //! Helper function to rebuild the index lists after the particles have been sorted
    void rebuildIndexList();

    //! Helper function to rebuild internal arrays
    void checkRebuild()
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
    void updateGPUAdvice();

    //! Helper function to be called when particles are added/removed
    void slotGlobalParticleNumChange()
        {
        m_global_ptl_num_change = true;
        }

    //! Helper function to build the 1:1 hash for tag membership
    void buildTagHash();

#ifdef ENABLE_HIP
    //! Helper function to rebuild the index lists after the particles have been sorted
    void rebuildIndexListGPU();
#endif
    };

namespace detail
    {
//! Exports the ParticleGroup class to python
void export_ParticleGroup(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
