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

/*! \file RigidData.cc
 	\brief Defines RigidData and related classes.
*/

#include "RigidData.h"
#include <cassert>

using namespace boost;
using namespace std;

/*! \param particle_data ParticleData this use in initializing this RigidData

	\pre \a particle_data has been completeley initialized with all arrays filled out
	\post All data members in RigidData are completely initialized from the given info in \a particle_data
*/
RigidData::RigidData(boost::shared_ptr<ParticleData> particle_data)
	: m_pdata(particle_data)
	{
	// read the body array from particle_data and extract some information
	
	// initialization should set this to true if no rigid bodies were defined in the particle data
	bool no_rigid_bodies_defined = false;
	if (no_rigid_bodies_defined)
		{
		// stop now and leave this class as an uninitialized shell
		m_n_bodies = 0;
		return;
		}
	
	// initialize the number of bodies
	m_n_bodies = 10;		// 10 is placeholder value in template
	unsigned int nmax = 5;	// 5 is placeholder value in template
	
	// allocate memory via construct & swap to avoid the temporaries produced by the = operator
	GPUArray<Scalar4> moment_inertia(m_n_bodies, m_pdata->getExecConf());
	GPUArray<unsigned int> body_size(m_n_bodies, m_pdata->getExecConf());
	GPUArray<unsigned int> particle_tags(nmax, m_n_bodies, m_pdata->getExecConf());
	GPUArray<unsigned int> particle_indices(nmax, m_n_bodies, m_pdata->getExecConf());
	GPUArray<Scalar4> particle_pos(nmax, m_n_bodies, m_pdata->getExecConf());

	GPUArray<Scalar4> com(m_n_bodies, m_pdata->getExecConf());
	GPUArray<Scalar4> vel(m_n_bodies, m_pdata->getExecConf());
	GPUArray<Scalar4> orientation(m_n_bodies, m_pdata->getExecConf());
	GPUArray<Scalar4> angmom(m_n_bodies, m_pdata->getExecConf());
	GPUArray<Scalar4> angvel(m_n_bodies, m_pdata->getExecConf());
	
	// swap the allocated GPUArray with the member variables
	m_moment_inertia.swap(moment_inertia);
	m_body_size.swap(body_size);
	m_particle_tags.swap(particle_tags);
	m_particle_indices.swap(particle_indices);
	m_particle_pos.swap(particle_pos);
		
	m_com.swap(com);
	m_vel.swap(vel);
	m_orientation.swap(orientation);
	m_angmom.swap(angmom);
	m_angvel.swap(angmom);
	
	// initialize the data
	initializeData();
	
	// initialize the index cace
	recalcIndices();
	
	// connect the sort signal
	m_sort_connection = m_pdata->connectParticleSort(bind(&RigidData::recalcIndices, this));	
	}

RigidData::~RigidData()
	{
	m_sort_connection.disconnect();
	}
		

/*!	\pre m_body_size has been filled with values
	\pre m_particle_tags hass been filled with values
	\pre m_particle_indices has been allocated
	\post m_particle_indices is updated to match the current sorting of the particle data
*/
void RigidData::recalcIndices()
	{
	// sanity check
	assert(m_pdata);
	assert(!m_particle_tags.isNull());
	assert(!m_particle_indices.isNull());
	assert(m_n_bodies <= m_particle_tags.getPitch());
	assert(m_n_bodies <= m_particle_indices.getPitch());
	assert(m_n_bodies == m_body_size.getNumElements());
	
	// get the particle data
	const ParticleDataArraysConst &arrays = m_pdata->acquireReadOnly();
	
	// get all the rigid data we need
	ArrayHandle<unsigned int> tags(m_particle_tags, access_location::host, access_mode::read);
	unsigned int tags_pitch = m_particle_tags.getPitch();
	
	ArrayHandle<unsigned int> indices(m_particle_indices, access_location::host, access_mode::readwrite);
	unsigned int indices_pitch = m_particle_indices.getPitch();
	
	ArrayHandle<unsigned int> body_size(m_body_size, access_location::host, access_mode::read);
	
	// for each body
	for (unsigned int body = 0; body < m_n_bodies; body++)
		{
		// for each particle in this body
		unsigned int len = body_size.data[body];
		assert(len <= m_particle_tags.getHeight() && len <= m_particle_indices.getHeight());
		for (unsigned int i = 0; i < len; i++)
			{
			// translate the tag to the current index
			unsigned int tag = tags.data[body*tags_pitch + i];
			unsigned int pidx = arrays.rtag[tag];
			indices.data[body*indices_pitch + i] = pidx;
			}
		}
		
	m_pdata->release();
	}
	
/*! \pre all data members have been allocated
	\post all data members are initialized with data from the particle data
*/
void RigidData::initializeData()
	{
	// template code does nothing here
	}
