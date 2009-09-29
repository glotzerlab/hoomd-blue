/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file ParticleGroup.h
    \brief Declares the ParticleGroup class
*/

#include <string>
#include <boost/shared_ptr.hpp>

#include "ParticleData.h"

#ifndef __PARTICLE_GROUP_H__
#define __PARTICLE_GROUP_H__

//! Describes a group of particles
/*! Some computations in HOOMD need to only be performed on certain groups of particles. ParticleGroup facilitates
    that by providing a flexible interface for choosing these groups that can be used by any other class in HOOMD.
    The most common use case is to iterate through all particles in the group, so the class will be optimized for
    that.

    Membership in the group can be determined by a range of particle types or tags. More complicated groups can be
    built up by taking unions and intersections of existing groups.

    There are potential issues with the particle type changing over the course of a simulation. Those issues are
    deferred for now. Groups will be evaluated on construction of the group and remain static for its lifetime.

    Another issue to the design is how to handle ParticleData? Groups may very well be used inside of a loop where
    the particle data has already been aquired, so ParticleGroup cannot hold onto a shared pointer and aquire again.
    It can only realistically aquire the data on contstruction.

    Pulling all these issue together, the best data structure to represent the group is to determine group membership
    on construction and generate a list of particle tags that belong to the group. In this way, iteration through the
    group is efficient and there is no dependance on accessing the ParticleData within the iteration.

    \ingroup data_structs
*/
class ParticleGroup
    {
    public:
        //! Option to pass to ParticleGroup::ParticleGroup() on construction to choose which criteria to apply
        enum criteriaOption
            {
            type,   //!< Select particles in the group by type
            tag     //!< Select particles in the group by tag
            };
            
        //! Constructs an empty particle group
        ParticleGroup() {};
        
        //! Constructs a particle group of all particles with the given criteria
        ParticleGroup(boost::shared_ptr<ParticleData> pdata, criteriaOption criteria, unsigned int min, unsigned int max);
        
        //! Get the number of members in the group
        /*! \returns The number of particles that belong to this group
        */
        const unsigned int getNumMembers() const
            {
            return (unsigned int)m_members.size();
            }
            
        //! Get a member from the group
        /*! \param i Index from 0 to getNumMembers()-1 of the group member to get
            \returns Tag of the member at index \a i
        */
        const unsigned int getMemberTag(unsigned int i) const
            {
            assert(i < getNumMembers());
            return m_members[i];
            }
            
        //! Make a new particle group from a union of two
        static boost::shared_ptr<ParticleGroup> groupUnion(boost::shared_ptr<ParticleGroup> a, boost::shared_ptr<ParticleGroup> b);
        //! Make a new particle group from an intersection
        static boost::shared_ptr<ParticleGroup> groupIntersection(boost::shared_ptr<ParticleGroup> a, boost::shared_ptr<ParticleGroup> b);
        
        
    private:
        std::vector<unsigned int> m_members;    //!< Lists the tags of the paritcle members
    };

//! Exports the ParticleGroup class to python
void export_ParticleGroup();

#endif
