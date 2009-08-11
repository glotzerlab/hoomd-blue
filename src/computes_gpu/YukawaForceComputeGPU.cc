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

// $Id: YukawaForceComputeGPU.cc 1368 2008-10-15 15:57:40Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes_gpu/YukawaForceComputeGPU.cc $

/*! \file YukawaForceComputeGPU.cc
	\brief Defines the YukawaForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "YukawaForceComputeGPU.h"
#include "cuda_runtime.h"

#include <stdexcept>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

/*! \param sysdef System to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\param kappa Coefficient to use in calculating the force
	\post memory is allocated and the parameter epsilon is set to 0.0
	\note The YukawaForceComputeGPU does not own the Neighborlist, the caller should
		delete the neighborlist when done.
*/
YukawaForceComputeGPU::YukawaForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, Scalar kappa) 
	: YukawaForceCompute(sysdef, nlist, r_cut, kappa)
	{
	// can't run on the GPU if there aren't any GPUs in the execution configuration
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a YukawaForceComputeGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing YukawaForceComputeGPU");
		}
	
	if (m_ntypes > 44)
		{
		cerr << endl << "***Error! YukawaForceComputeGPU cannot handle " << m_ntypes << " types" << endl << endl;
		throw runtime_error("Error initializing YukawaForceComputeGPU");
		}
		
	// default block size is the highest performance in testing on different hardware
	// choose based on compute capability of the device
	cudaDeviceProp deviceProp;
	int dev;
	exec_conf.gpu[0]->call(bind(cudaGetDevice, &dev));	
	exec_conf.gpu[0]->call(bind(cudaGetDeviceProperties, &deviceProp, dev));
	if (deviceProp.major == 1 && deviceProp.minor < 2)
		m_block_size = 192;
	else if (deviceProp.major == 1 && deviceProp.minor < 4)
		m_block_size = 416;
	else
		{
		cout << "***Warning! Unknown compute " << deviceProp.major << "." << deviceProp.minor << " when tuning block size for YukawaForceComputeGPU" << endl;
		m_block_size = 416;
		}

	// allocate the coeff data on the GPU
	int nbytes = sizeof(float2)*m_pdata->getNTypes()*m_pdata->getNTypes();
	
	d_coeffs.resize(exec_conf.gpu.size());
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&d_coeffs[cur_gpu]), nbytes));
		assert(d_coeffs[cur_gpu]);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void *)d_coeffs[cur_gpu], 0, nbytes));
		}
	// allocate the coeff data on the CPU
	h_coeffs = new float[m_pdata->getNTypes()*m_pdata->getNTypes()];
	}
	

YukawaForceComputeGPU::~YukawaForceComputeGPU()
	{
	exec_conf.tagAll(__FILE__, __LINE__);	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		assert(d_coeffs[cur_gpu]);
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)d_coeffs[cur_gpu]));
		}
	delete[] h_coeffs;
	}
	
/*! \param block_size Size of the block to run on the device
	Performance of the code may be dependant on the block size run
	on the GPU. \a block_size should be set to be a multiple of 32.
	\todo error check value
*/
void YukawaForceComputeGPU::setBlockSize(int block_size)
	{
	m_block_size = block_size;
	}

/*! \post The parameter \a epsilon is set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
	\note \a epsilon are low level parameters used in the calculation. 
	
	Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
	is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
	for will have epsilon is set to 0.0.
	
	\param typ1 Specifies one type of the pair
	\param typ2 Specifies the second type of the pair
	\param epsilon First parameter used to calcluate forces
*/
void YukawaForceComputeGPU::setParams(unsigned int typ1, unsigned int typ2, Scalar epsilon)
	{
	assert(h_coeffs);
	if (typ1 >= m_ntypes || typ2 >= m_ntypes)
		{
		cerr << endl << "***Error! Trying to set Yukawa params for a non existant type! " << typ1 << "," << typ2 << endl << endl;
		throw runtime_error("YukawaForceComputeGpu::setParams argument error");
		}
	
	// set coeffs in both symmetric positions in the matrix
	h_coeffs[typ1*m_pdata->getNTypes() + typ2] = epsilon;
	h_coeffs[typ2*m_pdata->getNTypes() + typ1] = epsilon;
	
	int nbytes = sizeof(float)*m_pdata->getNTypes()*m_pdata->getNTypes();
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, d_coeffs[cur_gpu], h_coeffs, nbytes, cudaMemcpyHostToDevice));
	}
		
/*! \post The Yukawa forces are computed for the given timestep on the GPU. 
	The neighborlist's compute method is called to ensure that it is up to date
	before forces are computed.
 	\param timestep Current time step of the simulation
 	
 	Calls gpu_compute_yukawa_forces to do the dirty work.
*/
void YukawaForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile
	if (m_prof) m_prof->push(exec_conf, "Yukawa pair");
	
	// The GPU implementation CANNOT handle a half neighborlist, error out now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	if (third_law)
		{
		cerr << endl << "***Error! YukawaForceComputeGPU cannot handle a half neighborlist" << endl << endl;
		throw runtime_error("Error computing forces in YukawaForceComputeGPU");
		}
	
	// access the neighbor list, which just selects the neighborlist into the device's memory, copying
	// it there if needed
	vector<gpu_nlist_array>& nlist = m_nlist->getListGPU();

	// access the particle data
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_yukawa_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], box, nlist[cur_gpu], d_coeffs[cur_gpu], m_pdata->getNTypes(), m_r_cut * m_r_cut, m_kappa, m_block_size));
		}
	exec_conf.syncAll();
	
	m_pdata->release();
	
	// the force data is now only up to date on the gpu
	m_data_location = gpu;

	Scalar avg_neigh = m_nlist->estimateNNeigh();
	int64_t n_calc = int64_t(avg_neigh * m_pdata->getN());
	int64_t mem_transfer = m_pdata->getN() * (4 + 16 + 16) + n_calc * (4 + 20);
	int64_t flops = n_calc * (3+12+5+2+2+6+3+2+7);
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	}

void export_YukawaForceComputeGPU()
	{
	class_<YukawaForceComputeGPU, boost::shared_ptr<YukawaForceComputeGPU>, bases<YukawaForceCompute>, boost::noncopyable >
		("YukawaForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, Scalar, Scalar >())
		.def("setBlockSize", &YukawaForceComputeGPU::setBlockSize)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

