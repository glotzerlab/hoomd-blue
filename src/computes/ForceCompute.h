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

#include <boost/shared_ptr.hpp>

#include "Compute.h"

#ifdef USE_CUDA
#include "gpu_pdata.h"
#endif

/*! \file ForceCompute.h
	\brief Declares a generic force computation
*/

#ifndef __FORCECOMPUTE_H__
#define __FORCECOMPUTE_H__

//! Handy structure for passing the force arrays around
/*! \c fx, \c fy, \c fz have length equal to the number of particles and store the x,y,z 
	components of the force on that particle. These pointers are 256-byte aligned on CUDA 
	compilations, in the order fx, fy, fz.
*/
struct ForceDataArrays
	{
	//! Zeroes pointers
	ForceDataArrays();
	
	Scalar const * __restrict__ fx; //!< x-component of the force
	Scalar const * __restrict__ fy; //!< y-component of the force
	Scalar const * __restrict__ fz; //!< z-component of the force
	};

//! Defines an interface for computing forces on each particle
/*! Derived classes actually provide the implementation that computes the forces.
	This base class exists so that some other part of the code can have a list of 
	ForceComputes without needing to know what types of forces are being calculated.
	The base class also implements the CPU <-> GPU copies of the force data.
	
	Like with ParticleData forces are stored with contiguous x,y,z components on the CPU
	and interleaved ones on the GPU. Translation is done on the device in a staging area.
	\ingroup computes
*/
class ForceCompute : public Compute
	{
	public:
		//! Constructs the compute
		ForceCompute(boost::shared_ptr<ParticleData> pdata);
		
		//! Destructor
		virtual ~ForceCompute();
		
		//! Access the computed force data
		const ForceDataArrays& acquire();

		#ifdef USE_CUDA
		//! Access the computed force data on the GPU
		float4 *acquireGPU();
		#endif
		
		//! Computes the forces
		virtual void compute(unsigned int timestep);

	protected:
		Scalar * __restrict__ m_fx;	//!< x-component of the force
		Scalar * __restrict__ m_fy; //!< y-component of the force
		Scalar * __restrict__ m_fz; //!< z-component of the force
		int m_nbytes;	//!< stores the number of bytes of memory allocated
		
		ForceDataArrays m_arrays;	//!< Structure-of-arrays for quick returning via acquire

		#ifdef USE_CUDA
		//! Simple type for identifying where the most up to date particle data is
		enum DataLocation
			{
			cpu,	//!< Particle data was last modified on the CPU
			cpugpu,	//!< CPU and GPU contain identical data
			gpu		//!< Particle data was last modified on the GPU
			};

		DataLocation m_data_location;   //!< Where the neighborlist data currently lives
		float4 *m_d_forces;				//!< Storage location for forces on the device
		float *m_d_staging;			//!< Staging array where values are (un)interleaved
		unsigned int m_uninterleave_pitch;	//!< Remember the pitch between x,y,z,type in the uninterleaved data
		unsigned int m_single_xarray_bytes;	//!< Remember the number of bytes allocated for a single float array

		//! Helper function to move data from the host to the device
		void hostToDeviceCopy();
		//! Helper function to move data from the device to the host
		void deviceToHostCopy();
		
		#endif

		//! Actually perform the computation of the forces
		/*! This is pure virtual here. Sub-classes must implement this function
			\param timestep Current time step
		 */
		virtual void computeForces(unsigned int timestep)=0;
	private:
		Scalar *m_data;	//!< The pointer where the memory is actually allocated
		
		#ifdef USE_PYTHON
		//! export_ForceCompute() needs to be a friend to export protected members
		friend void export_ForceCompute();
		#endif
	};
	
#ifdef USE_PYTHON
//! Exports the ForceCompute class to python
void export_ForceCompute();
#endif
	
#endif
