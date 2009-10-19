/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include "ParticleGroup.h"

#include <boost/python.hpp>
#include <boost/bind.hpp>
using namespace boost::python;
using namespace boost;

#include <algorithm>
#include <iostream>
using namespace std;

/*! \file ParticleGroup.cc
    \brief Defines the ParticleGroup class
*/

/*! \param pdata Particle data to build the group from
    \param criteria Criteria to select particles by
    \param min First value in the range to include
    \param max Last value in the range to include

    Particles where criteria falls within the range [min,max] (inclusive) are added to the group.
*/
ParticleGroup::ParticleGroup(boost::shared_ptr<ParticleData> pdata, criteriaOption criteria,
                             unsigned int min, unsigned int max)
    : m_pdata(pdata), m_is_member(pdata->getN()), m_member_idx(pdata->getN(), pdata->getExecConf())
    {
    const ParticleDataArraysConst& arrays = pdata->acquireReadOnly();
    
    // make a quick check on the sanity of the input data
    if (max < min)
        cout << "***Warning! max < min specified when creating a ParticleGroup" << endl;
        
    // switch based on the given criteria
    if (criteria == type)
        {
        // perform an input check on the data
        if (max >= pdata->getNTypes())
            cout << "***Warning! Requesting for non-existant particle type to be added to a ParticleGroup" << endl;
            
        // for each particle in the data
        for (unsigned int tag = 0; tag < arrays.nparticles; tag++)
            {
            // identify the index of the current particle tag
            unsigned int idx = arrays.rtag[tag];
            
            // add the tag to the list if it matches the criteria
            if (arrays.type[idx] <= max && arrays.type[idx] >= min)
                m_member_tags.push_back(tag);
            }
        }
    else if (criteria == tag)
        {
        // perform an input check on the data
        if (max >= pdata->getN())
            {
            cerr << endl << "***Error! Cannot create a group with tags larger than the number of particles " << endl << endl;
            throw runtime_error("Error creating ParticleGroup");
            }
            
        // for each particle in the range
        for (unsigned int tag = min; tag <= max; tag++)
            {
            // add it to the list
            m_member_tags.push_back(tag);
            }
        }
    else
        {
        cerr << endl << "***Error! Invalid critera specified when creating a ParticleGroup" << endl << endl;
        throw runtime_error("Error creating ParticleGroup");
        }
        
    pdata->release();
    
    // now that the tag list is completely set up and all memory is allocated, rebuild the index list
    rebuildIndexList();
    
    // connect the rebuildIndexList method to be called whenever the particles are sorted
    m_sort_connection = m_pdata->connectParticleSort(bind(&ParticleGroup::rebuildIndexList, this));
    }

ParticleGroup::~ParticleGroup()
    {
    // disconnect the sort connection, but only if there was a particle data to connect it to in the first place
    if (m_pdata)
        m_sort_connection.disconnect();
    }

/*! \returns Total mass of all particles in the group
    \note This method aquires the ParticleData internally
*/
const Scalar ParticleGroup::getTotalMass() const
    {
    // grab the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    
    // loop  through all indices in the group and total the mass
    Scalar total_mass = 0.0;
    for (unsigned int i = 0; i < getNumMembers(); i++)
        {
        unsigned int idx = getMemberIndex(i);
        total_mass += arrays.mass[idx];
        }
    m_pdata->release();
    return total_mass;
    }
    
/*! \returns The center of mass of the group, in unwrapped coordinates
    \note This method aquires the ParticleData internally
*/
const Scalar3 ParticleGroup::getCenterOfMass() const
    {
    // grab the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    
    // grab the box dimensions
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // loop  through all indices in the group and compute the weighted average of the positions
    Scalar total_mass = 0.0;
    Scalar3 center_of_mass = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    for (unsigned int i = 0; i < getNumMembers(); i++)
        {
        unsigned int idx = getMemberIndex(i);
        Scalar mass = arrays.mass[idx];
        total_mass += mass;
        center_of_mass.x += mass * (arrays.x[idx] + Scalar(arrays.ix[idx]) * Lx);
        center_of_mass.y += mass * (arrays.y[idx] + Scalar(arrays.iy[idx]) * Ly);
        center_of_mass.z += mass * (arrays.z[idx] + Scalar(arrays.iz[idx]) * Lz);
        }
    center_of_mass.x /= total_mass;
    center_of_mass.y /= total_mass;
    center_of_mass.z /= total_mass;

    m_pdata->release();
    
    return center_of_mass;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains all the elements present in \a a and
    \a b
*/
boost::shared_ptr<ParticleGroup> ParticleGroup::groupUnion(boost::shared_ptr<ParticleGroup> a, boost::shared_ptr<ParticleGroup> b)
    {
    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup());
    
    // make the union
    insert_iterator< vector<unsigned int> > ii(new_group->m_member_tags, new_group->m_member_tags.begin());
    set_union(a->m_member_tags.begin(), a->m_member_tags.end(), b->m_member_tags.begin(), b->m_member_tags.end(), ii);
    
    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements present in both \a a and
    \a b
*/
boost::shared_ptr<ParticleGroup> ParticleGroup::groupIntersection(boost::shared_ptr<ParticleGroup> a, boost::shared_ptr<ParticleGroup> b)
    {
    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup());
    
    // make the union
    insert_iterator< vector<unsigned int> > ii(new_group->m_member_tags, new_group->m_member_tags.begin());
    set_intersection(a->m_member_tags.begin(), a->m_member_tags.end(), b->m_member_tags.begin(), b->m_member_tags.end(), ii);
    
    // return the newly created group
    return new_group;
    }


/*! \pre m_member_tags has been filled out, listing all particle tags in the group
    \pre memory has been allocated for m_is_member and m_member_idx
    \post m_is_member is updated so that it reflects the current indices of the particles in the group
    \post m_member_idx is updated listing all particle indices belonging to the group, in index order
*/
void ParticleGroup::rebuildIndexList()
    {
    // start by rebuilding the bitset of member indices in the group
    // it needs to be cleared first
    m_is_member.reset();
    
    // then loop through every particle in the group and set its bit
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    for (unsigned int member_idx = 0; member_idx < m_member_tags.size(); member_idx++)
        {
        unsigned int idx = arrays.rtag[m_member_tags[member_idx]];
        m_is_member[idx] = true;
        }
    m_pdata->release();
    
    // then loop through the bitset and add indices to the index list
    ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::readwrite);
    unsigned int cur_member = 0;
    for (unsigned int idx = 0; idx < arrays.nparticles; idx++)
        {
        if (isMember(idx))
            {
            h_handle.data[cur_member] = idx;
            cur_member++;
            }
        }
    
    // sanity check, the number of indices added to m_member_idx must be the same as the number of members in the group
    assert(cur_member == m_member_tags.size());
    }

void export_ParticleGroup()
    {
    scope in_particlegroup = class_<ParticleGroup, boost::shared_ptr<ParticleGroup>, boost::noncopyable>
                             ("ParticleGroup", init< boost::shared_ptr<ParticleData>, ParticleGroup::criteriaOption, unsigned int, unsigned int >())
                             .def("getNumMembers", &ParticleGroup::getNumMembers)
                             .def("getMemberTag", &ParticleGroup::getMemberTag)
                             .def("groupUnion", &ParticleGroup::groupUnion)
                             .def("groupIntersection", &ParticleGroup::groupIntersection)
                             ;
                             
    enum_<ParticleGroup::criteriaOption>("criteriaOption")
    .value("type", ParticleGroup::type)
    .value("tag", ParticleGroup::tag)
    ;
    }

