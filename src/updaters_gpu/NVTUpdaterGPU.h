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

/*! \file NVTUpdaterGPU.h
	\brief Declares the NVTUpdaterGPU class
*/

#include "NVTUpdater.h"
#include "NVTUpdaterGPU.cuh"

#include <boost/shared_ptr.hpp>

#ifndef __NVTUPDATER_GPU_H__
#define __NVTUPDATER_GPU_H__

//! NVT via Nose-Hoover on the GPU
/*!	NVTUpdaterGPU implements exactly the same caclulations as NVTUpdater, but on the GPU.
	
	The GPU kernel that accomplishes this can be found in gpu_nvt_kernel.cu

	\ingroup updaters
*/
class NVTUpdaterGPU : public NVTUpdater
	{
	public:
		//! Constructor
		NVTUpdaterGPU(boost::shared_ptr<ParticleData> pdata, Scalar deltaT, Scalar tau, Scalar T);
		virtual ~NVTUpdaterGPU();

		//! Take one timestep forward
		virtual void update(unsigned int timestep);
		
	private:
		std::vector<gpu_nvt_data> d_nvt_data;	//!< Temp data on the device needed to implement NVT

		//! Helper function to allocate data
		void allocateNVTData(int block_size);
		
		//! Helper function to free data
		void freeNVTData();
	};
	
//! Exports the NVTUpdater class to python
void export_NVTUpdaterGPU();

#endif
