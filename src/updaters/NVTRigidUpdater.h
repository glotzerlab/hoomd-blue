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

/*! \file NVEUpdater.h
	\brief Declares an updater that implements NVE dynamics
*/

#include "Variant.h"
#include "NVERigidUpdater.h"
#include <vector>
#include <boost/shared_ptr.hpp>

#ifndef __NVTRIGIDUPDATER_H__
#define __NVTRIGIDUPDATER_H__



class SystemDefinition;

//! Updates particle positions and velocities
/*! This updater performes constant N, constant volume, constant energy (NVE) dynamics. Particle positions and velocities are 
	updated according to the velocity verlet algorithm. The forces that drive this motion are defined external to this class
	in ForceCompute. Any number of ForceComputes can be given, the resulting forces will be summed to produce a net force on 
	each particle.
	
	\ingroup updaters
*/

class NVTRigidUpdater : public NVERigidUpdater
	{
	public:
		//! Constructor
		NVTRigidUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT, boost::shared_ptr<Variant> temperature);

		//! Setup the initial net forces, torques and angular momenta
		void setup();
		
		//! First step of velocit Verlet integration
		void initialIntegrate(unsigned int timestep);
		
		//! Second step of velocit Verlet integration
		void finalIntegrate(unsigned int timestep);
	
	
		const GPUArray<Scalar>& getEtaDotT() { return eta_dot_t; }
		const GPUArray<Scalar>& getEtaDotR() { return eta_dot_r; }
		const GPUArray<Scalar4>& getConjqm() { return conjqm; }

		void updateThermostats(Scalar akin_t, Scalar akin_r, unsigned int timestep) { update_nhcp(akin_t, akin_r, timestep); }

	protected:
		//! Private member functions using parameters to avoid duplicate array handles declaration
		void update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep);
		void no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt);
		void quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
		void inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c);
		void matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
		void transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c);
		inline Scalar maclaurin_series(Scalar x);

		boost::shared_ptr<Variant> m_temperature;
		Scalar boltz, t_freq;
		Scalar nf_t, nf_r;
		unsigned int chain, iter, order;

		GPUArray<Scalar>	q_t;
		GPUArray<Scalar>	q_r;
		GPUArray<Scalar>	eta_t;
		GPUArray<Scalar>	eta_r;
		GPUArray<Scalar>	eta_dot_t;
		GPUArray<Scalar>	eta_dot_r;
		GPUArray<Scalar>	f_eta_t;
		GPUArray<Scalar>	f_eta_r;

		GPUArray<Scalar>	w;
		GPUArray<Scalar>	wdti1;
		GPUArray<Scalar>	wdti2;
		GPUArray<Scalar>	wdti4;
		GPUArray<Scalar4>	conjqm;
		
		

	};
	
	
#endif
