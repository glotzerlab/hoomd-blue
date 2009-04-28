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

/*! \file RigidData.h
 	\brief Contains declarations for RigidData and related classes.
*/

#ifndef __RIGID_DATA_H__
#define __RIGID_DATA_H__

#include "ParticleData.h"
#include "GPUArray.h"

//! Stores all per rigid body values
/*! All rigid body data (except for the per-particle body value) is stored in RigidData
	which can be accessed from SystemDefinition. On construction, RigidData will read the body
	tag from the passed in ParticleData and initialize all rigid body data structures.
	
	The 2D arrays in this class bear a little explanation. They are arranged as \a n_max by 
	\b n_bodies arrays (where \a n_max is the size of the largest rigid body in the system). Bodies
	are listed across the column and the quantity for each particle in that body is listed down the
	rows of the corresponding column. Thus, to access the index of particle \a p in body \b b, one
	would access array element \c particle_indices_handle.data[b*pitch+p] . This will set us up for 
	the fastest GPU implementation of tasks like summing the force/torque on each body as we will 
	be able to process 1 body in each block with one particle in each thread, performing any sums as
	reductions.
	
	\ingroup data_structs
*/
class RigidData
	{
	public:
		//! Initializes all rigid body data from the given particle data
		RigidData(boost::shared_ptr<ParticleData> particle_data);
		//! Destructor
		~RigidData();
		
		//! Get the number of bodies in the rigid data
		unsigned int getNumBodies() { return m_n_bodies; }
		
		//! \name getter methods (static data)
		//@{
		//! Get the m_moment_inertial
		const GPUArray<Scalar4>& getMomentInertia() { return m_moment_inertia; }
		//! Get m_body_size
		const GPUArray<unsigned int>& getBodySize() { return m_body_size; }
		//! Get the m_particle_tags
		const GPUArray<unsigned int>& getParticleTags() { return m_particle_tags; }
		//! Get m_particle_indices
		const GPUArray<unsigned int>& getParticleIndices() { return m_particle_indices; }
		//! Get m_particle_pos
		const GPUArray<Scalar4>& getParticlePos() { return m_particle_pos; }
		//! Get m_mass
		const GPUArray<Scalar>& getBodyMass() { return m_body_mass; }
		//@}
		
		//! \name getter methods (integrated data)
		//@{
		//! Get m_com
		const GPUArray<Scalar4>& getCOM() { return m_com; }
		//! Get m_vel
		const GPUArray<Scalar4>& getVel() { return m_vel; }
		//! Get m_orientation
		const GPUArray<Scalar4>& getOrientation() { return m_orientation; }
		//! Get m_angmom
		const GPUArray<Scalar4>& getAngMom() { return m_angmom; }
		//! Get m_angvel
		const GPUArray<Scalar4>& getAngVel() { return m_angvel; }
		//! Get m_body_imagex
		const GPUArray<unsigned int>& getBodyImagex() { return m_body_imagex; }
		//! Get m_body_imagey
		const GPUArray<unsigned int>& getBodyImagey() { return m_body_imagey; }
		//! Get m_body_imagez
		const GPUArray<unsigned int>& getBodyImagez() { return m_body_imagez; }
		//! Get m_ex_space
		const GPUArray<Scalar4>& getExSpace() { return m_ex_space; }
		//! Get m_ey_space
		const GPUArray<Scalar4>& getEySpace() { return m_ey_space; }
		//! Get m_ez_space
		const GPUArray<Scalar4>& getEzSpace() { return m_ez_space; }
		//@}
		
		//! Intitialize and fill out all data members: public to be called from NVEUpdater when the body information of particles wss already set.
		void initializeData();

	private:
		boost::shared_ptr<ParticleData> m_pdata;		//!< The particle data with which this RigidData is associated
		boost::signals::connection m_sort_connection;	//!< Connection to the resort signal from ParticleData
		
		//! \name static data members (set on initialization)
		//@{
		unsigned int m_n_bodies;					//!< Number of rigid bodies in the data structure
		GPUArray<Scalar4> m_moment_inertia;			//!< n_bodies length 1D array of moment of interias in the body frame
		GPUArray<unsigned int> m_body_size;			//!< n_bodies length 1D array listing the size of each rigid body
		GPUArray<unsigned int> m_particle_tags;		//!< n_max by n_bodies 2D array listing particle tags belonging to bodies
		GPUArray<unsigned int> m_particle_indices;	//!< n_max by n_bodies 2D array listing particle indices belonging to bodies (updated when particles are resorted)
		GPUArray<Scalar4> m_particle_pos;			//!< n_max by n_bodies 2D array listing particle positions relative to the COM for this body in which body-fixed frame
		GPUArray<Scalar> m_body_mass;				//!< n_bodies length 1D array of body mass
		//@}
		
		//! \name dynamic data members (updated via integration)
		//@{ 
		GPUArray<Scalar4> m_com;			//!< n_bodies length 1D array of center of mass positions
		GPUArray<Scalar4> m_vel;			//!< n_bodies length 1D array of body velocities
		GPUArray<Scalar4> m_orientation;	//!< n_bodies length 1D array of orientation quaternions
		GPUArray<Scalar4> m_angmom;			//!< n_bodies length 1D array of angular momentum in the space frame
		GPUArray<Scalar4> m_angvel;			//!< n_bodies length 1D array of angular velocity in the space frame
		
		GPUArray<unsigned int> m_body_imagex;	//!< n_bodies length 1D array of the body image in x direction
		GPUArray<unsigned int> m_body_imagey;	//!< n_bodies length 1D array of the body image in y direction
		GPUArray<unsigned int> m_body_imagez;	//!< n_bodies length 1D array of the body image in z direction
		GPUArray<Scalar4> m_ex_space;		//!< n_bodies length 1D array of the x axis of the body frame in the space frame
		GPUArray<Scalar4> m_ey_space;		//!< n_bodies length 1D array of the y axis of the body frame in the space frame
		GPUArray<Scalar4> m_ez_space;		//!< n_bodies length 1D array of the z axis of the body frame in the space frame
		
		//@}

		//! Recalculate the cached indices from the stored tags after a particle sort
		void recalcIndices();
		
		//! Functions used to diagonalize the inertia tensor for moment inertia and principle axes
		int diagonalize(Scalar **matrix, Scalar *evalues, Scalar **evectors);
		void rotate(Scalar **matrix, int i, int j, int k, int l, Scalar s, Scalar tau);
		void quaternionFromExyz(Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space, Scalar4 &quat);
		
	};
	
#endif
