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

#include "ParticleGroup.h"

#include <boost/python.hpp>
using namespace boost::python;

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
				m_members.push_back(tag);
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
			m_members.push_back(tag);
			}
		}
	else
		{
		cerr << endl << "***Error! Invalid critera specified when creating a ParticleGroup" << endl << endl;
		throw runtime_error("Error creating ParticleGroup");	
		}
	
	pdata->release();
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
	insert_iterator< vector<unsigned int> > ii(new_group->m_members, new_group->m_members.begin());
	set_union(a->m_members.begin(), a->m_members.end(), b->m_members.begin(), b->m_members.end(), ii);
	
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
	insert_iterator< vector<unsigned int> > ii(new_group->m_members, new_group->m_members.begin());
	set_intersection(a->m_members.begin(), a->m_members.end(), b->m_members.begin(), b->m_members.end(), ii);
	
	// return the newly created group
	return new_group;
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

