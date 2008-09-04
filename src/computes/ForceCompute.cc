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

/*! \file ForceCompute.cc
	\brief Defines the ForceCompute class
*/

#include "ForceCompute.h"
#include <iostream>
using namespace std;

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
using namespace boost;

/*! \post \c fx, \c fy, \c fz are all set to NULL
*/
ForceDataArrays::ForceDataArrays() : fx(NULL), fy(NULL), fz(NULL)
	{
	}
	
/*! \param pdata ParticleData to compute forces on
	\post The Compute is initialized and all memory needed for the forces is allocated
	\post \c fx, \c fy, \c fz pointers in m_arrays are set
	\post All forces are initialized to 0
*/
ForceCompute::ForceCompute(boost::shared_ptr<ParticleData> pdata) : Compute(pdata), m_particles_sorted(false)
	{
	assert(pdata);
	assert(pdata->getN());
	
	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	#endif	
				
	// allocate the memory here in the same way as with the ParticleData: put all 3
	// arrays back to back. 256-byte align the data so that uninterleaved <-> interleaved
	// translation can be done easily
	#ifdef USE_CUDA
	// start by adding up the number of bytes needed for the Scalar arrays, rounding up by 256
	unsigned int single_xarray_bytes = sizeof(Scalar) * pdata->getN();
	if ((single_xarray_bytes & 255) != 0)
		single_xarray_bytes += 256 - (single_xarray_bytes & 255);
	
	// total all bytes from scalar arrays
	m_nbytes = single_xarray_bytes * 5;
	
	if (!exec_conf.gpu.empty())
		{
		exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[0]->call(bind(cudaMallocHost, (void **)((void *)&m_data), m_nbytes));	
		}
	else
		m_data = (Scalar *)malloc(m_nbytes);
	#else
	// start by adding up the number of bytes needed for the Scalar arrays
	unsigned int single_xarray_bytes = sizeof(Scalar) * pdata->getN();
	
	// total all bytes from scalar arrays
	m_nbytes = single_xarray_bytes * 5;
	m_data = (Scalar *)malloc(m_nbytes);
	#endif

	assert(m_data);

	// Now that m_data is allocated, we need to play some pointer games to assign
	// the x,y,z, etc... pointers
	char *cur_byte = (char *)m_data;
	m_arrays.fx = m_fx = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays.fy = m_fy = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays.fz = m_fz = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays.pe = m_pe = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	m_arrays.virial = m_virial = (Scalar *)cur_byte;  cur_byte += single_xarray_bytes;
	
	// should be good to go now
	assert(m_fx);
	assert(m_fy);
	assert(m_fz);
	assert(m_pe);
	assert(m_arrays.fx);
	assert(m_arrays.fy);
	assert(m_arrays.fz);
	assert(m_arrays.pe);
	
	// zero the data
	memset((void*)m_data, 0, m_nbytes);
	
	#ifdef USE_CUDA
	// make space for all the memory pointers
	m_d_forces.resize(exec_conf.gpu.size());
	m_d_staging.resize(exec_conf.gpu.size());
	
	// allocate device memory for the forces and staging memory
	m_uninterleave_pitch = single_xarray_bytes/4;
	m_single_xarray_bytes = single_xarray_bytes;

	if (!exec_conf.gpu.empty())
		{
		cout << "***Warning! Virial data structure not yet implemented on the GPU" << endl;
		
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_forces[cur_gpu]), single_xarray_bytes*4) );
			exec_conf.gpu[cur_gpu]->call(bind(cudaMalloc, (void **)((void *)&m_d_staging[cur_gpu]), single_xarray_bytes*4) );
			}
		
		m_h_staging = new float4[pdata->getN()];
		
		deviceToHostCopy();
		m_data_location = cpugpu;
		}
	else
		m_data_location = cpu;
	#endif

	// connect to the ParticleData to recieve notifications when particles change order in memory
	m_sort_connection = m_pdata->connectParticleSort(bind(&ForceCompute::setParticlesSorted, this));
	}	
	
/*! Frees allocated memory
*/
ForceCompute::~ForceCompute()
	{
	assert(m_data);
		
	// free the data, which needs to be condtionally compiled for CUDA and non CUDA builds
	#ifdef USE_CUDA
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (!exec_conf.gpu.empty())
		exec_conf.gpu[0]->call(bind(cudaFreeHost, m_data));
	else
		free(m_data);
	#else
	free(m_data);
	#endif
	
	m_data = NULL;
	m_arrays.fx = m_fx = NULL;
	m_arrays.fy = m_fy = NULL;
	m_arrays.fz = m_fz = NULL;
	m_arrays.pe = m_pe = NULL;
	
	#ifdef USE_CUDA
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{	
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_d_forces[cur_gpu]));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_d_staging[cur_gpu]));
		}
	
	if (!exec_conf.gpu.empty())
		delete[] m_h_staging;
	#endif

	m_sort_connection.disconnect();
	}
	
/*! Sums the total potential energy calculated by the last call to compute() and returns it.
*/
Scalar ForceCompute::calcEnergySum()
	{
	const ForceDataArrays& arrays = acquire();
	
	// always perform the sum in double precision for better accuracy
	// this is cheating and is really just a temporary hack to get logging up and running
	// the potential accuracy loss in simulations needs to be evaluated here and a proper
	// summation algorithm put in place
	double pe_total = 0.0;
	for (unsigned int i=0; i < m_pdata->getN(); i++)
		{
		pe_total += (double)arrays.pe[i];
		}

	return Scalar(pe_total);
	}

/*! Access the computed forces on the CPU, this may require copying data from the GPU
 	\returns Structure of arrays of the x,y,and z components of the forces on each particle
 			calculated by the last call to compute()
 	\note These are const pointers so the caller cannot muss with the data
 */
const ForceDataArrays& ForceCompute::acquire()
	{
	#ifdef USE_CUDA

	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is solely on the cpu, life is easy, return the data arrays
			// and stay in the same state
			return m_arrays;
			break;
		case cpugpu:
			// if the data is up to date on both the cpu and gpu, life is easy, return
			// the data arrays and stay in the same state
			return m_arrays;
			break;
		case gpu:
			// if the data resides on the gpu, it needs to be copied back to the cpu
			// this changes to the cpugpu state since the data is now fully up to date on 
			// both
			deviceToHostCopy();
			m_data_location = cpugpu;
			return m_arrays;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_arrays;	
			break;
        }

    #else

	return m_arrays;
	#endif
	}

#ifdef USE_CUDA
/*! Access computed forces on the GPU. This may require copying data from the CPU if the forces
	were computed there.
	\returns Data pointer to the forces on the GPU

	\note For performance reasons, the returned pointer will \b not change
	from call to call. The call still must be made, however, to ensure that
	the data has been copied to the GPU.
*/
vector<float4 *>& ForceCompute::acquireGPU()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	if (exec_conf.gpu.empty())
		{
		cerr << endl << "***Error! Acquiring forces on GPU, but there is no GPU in the exection configuration" << endl << endl;
		throw runtime_error("Error acquiring GPU forces");
		}
	
	// this is the complicated graphics card version, need to do some work
	// switch based on the current location of the data
	switch (m_data_location)
		{
		case cpu:
			// if the data is on the cpu, we need to copy it over to the gpu
			hostToDeviceCopy();
			// now we are in the cpugpu state
			m_data_location = cpugpu;
			return m_d_forces;
			break;
		case cpugpu:
			// if the data is up to date on both the cpu and gpu, life is easy
			// state remains the same, and return it
			return m_d_forces;
			break;
		case gpu:
			// if the data resides on the gpu, life is easy
			// state remains the same, and return it     
			return m_d_forces;
			break;
		default:
			// anything other than the above is an undefined state!
			assert(false);
			return m_d_forces;	
			break;		
		}
	}

/*! The data copy is performed efficiently by transferring data as it is on the CPU
	(256 byte aligned arrays) and then interleaving it on the GPU
*/
void ForceCompute::hostToDeviceCopy()
	{
	if (m_prof) m_prof->push("ForceCompute - CPU->GPU");
	
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy force data to the staging area
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_d_staging[cur_gpu], m_data, m_single_xarray_bytes*4, cudaMemcpyHostToDevice));
		// interleave the data
		exec_conf.gpu[cur_gpu]->call(bind(gpu_interleave_float4, m_d_forces[cur_gpu], m_d_staging[cur_gpu], m_pdata->getN(), m_uninterleave_pitch));
		}
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(0, m_single_xarray_bytes*4);
		}
	}

/*! \sa hostToDeviceCopy()
*/
void ForceCompute::deviceToHostCopy()
	{
	if (m_prof) m_prof->push("ForceCompute - GPU->CPU");

	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		// get local particle indices
		unsigned int local_beg = m_pdata->getLocalBeg(cur_gpu);
		unsigned int local_num = m_pdata->getLocalNum(cur_gpu);
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		// copy force data
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_h_staging, m_d_forces[cur_gpu] + local_beg, local_num*sizeof(float4), cudaMemcpyDeviceToHost));
		// fill out host forces
		for (unsigned int i = 0; i < local_num; i++)
			{
			m_fx[local_beg + i] = m_h_staging[i].x;
			m_fy[local_beg + i] = m_h_staging[i].y;
			m_fz[local_beg + i] = m_h_staging[i].z;
			m_pe[local_beg + i] = m_h_staging[i].w;
			}
		}
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(0, m_single_xarray_bytes*4);
		}
	}

#endif
	
/*! Performs the force computation.
	\param timestep Current Timestep
	\note If compute() has previously been called with a value of timestep equal to
		the current value, the forces are assumed to already have been computed and nothing will 
		be done
*/
void ForceCompute::compute(unsigned int timestep)
	{
	// skip if we shouldn't compute this step
	if (!m_particles_sorted && !shouldCompute(timestep))
		return;
		
	computeForces(timestep);
	m_particles_sorted = false;
	}
	
#ifdef USE_PYTHON

//! Wrapper class for wrapping pure virtual methodos of ForceCompute in python
class ForceComputeWrap : public ForceCompute, public wrapper<ForceCompute>
	{
	public:
		//! Constructor
		/*! \param pdata Particle data passed to the base class */
		ForceComputeWrap(shared_ptr<ParticleData> pdata) : ForceCompute(pdata) { }
	protected:
		//! Calls the overidden ForceCompute::computeForces()
		/*! \param timestep parameter to pass on to the overidden method */
		void computeForces(unsigned int timestep)
			{
			this->get_override("computeForces")(timestep);
			}
	};

// a decision has been made to not support python classes derived from force computes at this time:
// only export the public interface

void export_ForceCompute()
	{
	class_< ForceComputeWrap, boost::shared_ptr<ForceComputeWrap>, bases<Compute>, boost::noncopyable >
		("ForceCompute", init< boost::shared_ptr<ParticleData> >())
		.def("acquire", &ForceCompute::acquire, return_value_policy<copy_const_reference>())
		//.def("computeForces", pure_virtual(&ForceCompute::computeForces))
		;
	}
#endif
