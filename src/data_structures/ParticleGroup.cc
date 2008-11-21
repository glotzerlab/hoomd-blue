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

#include "ParticleGroup.h"

#include <boost/python.hpp>
using namespace boost::python;

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
			cerr << endl << "***Error! Cannot create a group with tags larger than the number of particles " << endl;
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
		cerr << endl << "***Error! Invalid critera specified when creating a ParticleGroup" << endl;
		throw runtime_error("Error creating ParticleGroup");	
		}
	
	pdata->release();
	}
	
void export_ParticleGroup()
	{
	scope in_particlegroup = class_<ParticleGroup, boost::shared_ptr<ParticleGroup>, boost::noncopyable>
		("ParticleGroup", init< boost::shared_ptr<ParticleData>, ParticleGroup::criteriaOption, unsigned int, unsigned int >())
		.def("getNumMembers", &ParticleGroup::getNumMembers)
		.def("getMemberTag", &ParticleGroup::getMemberTag)
		;
		
	enum_<ParticleGroup::criteriaOption>("criteriaOption")
		.value("type", ParticleGroup::type)
		.value("tag", ParticleGroup::tag)
		;
	}

