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

// $Id: NVTUpdaterGPU.h 1325 2008-10-06 13:42:07Z joaander $
// $URL: https://svn2.assembla.com/svn/hoomd/trunk/src/updaters_gpu/NVTUpdaterGPU.h $

/*! \file NVTUpdaterGPU.h
	\brief Declares the NVTUpdaterGPU class
*/

#include "NPTUpdater.h"
#include "NPTUpdaterGPU.cuh"

#include <boost/shared_ptr.hpp>

#ifndef __NPTUPDATER_GPU_H__
#define __NPTUPDATER_GPU_H__

//! NPT
/*! \ingroup updaters
*/
class NPTUpdaterGPU : public NPTUpdater
	{
	public:
		//! Constructor
	        NPTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar tauP, Scalar T, Scalar P);
		virtual ~NPTUpdaterGPU();

		//! Take one timestep forward
		virtual void update(unsigned int timestep);

		//! Overides addForceCompute to add virial computes
		virtual void addForceCompute(boost::shared_ptr<ForceCompute> fc);
				
		//! overides removeForceCompute to remove all virial computes
		virtual void removeForceComputes();
	
		//! Computes current pressure
		virtual Scalar computePressure(unsigned int timestep);
		
		//! Computes current temperature
		virtual Scalar computeTemperature(unsigned int timestep);

	private:
		std::vector<gpu_npt_data> d_npt_data;	//!< Temp data on the device needed to implement NPT

		//! Helper function to allocate data
		void allocateNPTData(int block_size);
		
		//! Helper function to free data
		void freeNPTData();

		//! Virial data pointers on the device
		vector<float **> m_d_virial_data_ptrs;
	};
	
//! Exports the NPTUpdater class to python
void export_NPTUpdaterGPU();

extern "C" cudaError_t integrator_sum_virials(gpu_pdata_arrays *pdata, float** virial_list, int num_virials, gpu_npt_data* nptdata);


#endif
