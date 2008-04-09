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

#include "NeighborListNsqGPU.h"

#include <iostream>
using namespace std;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

/*! \param pdata Particle data the neighborlist is to compute neighbors for
	\param r_cut Cuttoff radius under which particles are considered neighbors
	\param r_buff Buffer distance around the cuttoff within which particles are included
	\post NeighborList is initialized and the list memory has been allocated,
		but the list will not be computed until compute is called.
	\post The storage mode defaults to half
	\todo update docs
*/
NeighborListNsqGPU::NeighborListNsqGPU(boost::shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff) 
	: NeighborList(pdata, r_cut, r_buff)
	{
	m_storage_mode = full;
	}

NeighborListNsqGPU::~NeighborListNsqGPU()
	{
	}
		 
void NeighborListNsqGPU::compute(unsigned int timestep)
	{
	// skip if we shouldn't compute this step
	if (!shouldCompute(timestep) && !m_force_update)
		return;

	if (m_storage_mode != full)
		{
		cerr << "Only full mode nlists can be generated on the GPU" << endl;
		exit(1);
		}
	
	if (m_prof)
        m_prof->push("Nlist^2.GPU");
		
	// need to update the exclusion data if anything has changed
	if (m_force_update)
		updateExclusionData();
		
	if (needsUpdating(timestep))
		{
		// access the particle data
		gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
		gpu_boxsize box = m_pdata->getBoxGPU();

		// create a temporary copy of r_max sqaured
   		Scalar r_max_sq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);			
		
    	if (m_prof)
        	m_prof->push("Build list");
		
		// calculate the nlist
		gpu_nlist_nsq(&pdata, &box, &m_gpu_nlist, r_max_sq);

		m_data_location = gpu;
    
		// amount of memory transferred is N * 16 + N*N*16 of particle data / number of threads in a block. We'll ignore the nlist data for now
		int64_t mem_transfer = int64_t(m_pdata->getN())*16 + int64_t(m_pdata->getN())*int64_t(m_pdata->getN())*16 / 128;
		// number of flops is 21 for each N*N
		int64_t flops = int64_t(m_pdata->getN())*int64_t(m_pdata->getN())*21;
	
		m_pdata->release();
	
		if (m_prof)
			{
			cudaThreadSynchronize();
        	m_prof->pop(flops, mem_transfer);
			}
		}
		
	if (m_prof)
        m_prof->pop();	
	}
 
#ifdef USE_PYTHON
void export_NeighborListNsqGPU()
	{
	class_<NeighborListNsqGPU, boost::shared_ptr<NeighborListNsqGPU>, bases<NeighborList>, boost::noncopyable >
		("NeighborListNsqGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		;
	}
#endif
