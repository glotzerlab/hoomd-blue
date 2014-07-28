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

// Maintainer: joaander

/*! \file RigidBodyGroup.h
    \brief Declares the RigidBodyGroup class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include "ParticleGroup.h"

#ifndef __RIGID_BODY_GROUP_H__
#define __RIGID_BODY_GROUP_H__

//! Describes a group of rigid bodies
/*! \b Overview

    Some computations in HOOMD need to only be performed on certain groups of rigid bodies. Such groups are initialized
    from an already existing ParticleGroup for simplicity and integration into the rest of HOOMD.

    The following common use-cases are expected and the design is tuned to make these optimal.
     - Iterate through all rigid body indices in the group, and the order of iteration doesn't matter, except for
       performance.
     - O(1) test if a particular body index is in the group

    Group membership is determined once at the instantiation of the group. Thus RigidBodyGroup only supports static
    groups where membership does not change over the course of a simulation. Dynamic groups, if they are needed,
    may require a drastically different design to allow for efficient access.

    Membership is determined based on the particle members in a ParticleGroup. If only a portion of a body is included
    in the ParticleGroup, the body is included, and a warning is also issue to the user that a body has been partially
    selected.

    In many use-cases, RigidBodyGroup may be accessed many times within inner loops. Thus, it must not aquire any
    ParticleData arrays within most of the get() calls as the caller must be allowed to leave their ParticleData
    aquired. Thus, all get() methods must return values from internal cached variables only. Those methods that
    absolutely require the particle data be released before they are called will be documented as such.

    <b>Data Structures and Implementation</b>

    The initial and fundamental data structure in the group is a vector listing all of the bodies in the group,
    in a sorted index order. This list can be accessed directly via getMemberIndex() to meet the 1st use case listed
    above. A dynamic bitset is used to store one bit per rigid body for efficient O(1) tests if a given body is in
    the group.

    Finally, the common use case on the GPU using groups will include threads that access the group membership.
    For that it needs a list of indices of all the bodies in the group. To facilitates this, the list of indices
    in the group will be stored in a GPUArray.

    \ingroup data_structs
*/
class RigidBodyGroup
    {
    public:
        //! \name Initialization methods
        // @{

        //! Constructs a rigid body groupp given a particle group
        RigidBodyGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group);

        // @}
        //! \name Accessor methods
        // @{

        //! Get the number of members in the group
        /*! \returns The number of particles that belong to this group
        */
        unsigned int getNumMembers() const
            {
            return (unsigned int)m_member_idx.getNumElements();
            }

        //! Get a member index from the group
        /*! \param j Value from 0 to getNumMembers()-1 of the group member to get
            \returns Index of the member at position \a j
            \note Internally acquires the GPUArray from getIndexArray
        */
        unsigned int getMemberIndex(unsigned int j) const
            {
            assert(j < getNumMembers());
            ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::read);
            return h_handle.data[j];
            }

        //! Test if a body index is a member of the group
        /*! \param idx Index of the body to query (from 0 to the number of bodies in RigidData -1)
            \returns true if the body with index \a idx is in the group
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

    private:
        boost::shared_ptr<SystemDefinition> m_sysdef;   //!< The system definition this group is associated with
        boost::shared_ptr<RigidData> m_rdata;           //!< The system definition this group is associated with
        boost::shared_ptr<ParticleData> m_pdata;        //!< The particle data this group is associated with
        boost::dynamic_bitset<> m_is_member;            //!< One bit per particle, true if index is a member of the group
        GPUArray<unsigned int> m_member_idx;            //!< List of all particle indices in the group

        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
    };

#endif
