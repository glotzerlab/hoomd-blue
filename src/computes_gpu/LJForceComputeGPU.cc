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

/*! \file LJForceComputeGPU.cc
	\brief Defines the LJForceComputeGPU class
*/

#include "LJForceComputeGPU.h"
#include "cuda_runtime.h"

#include <stdexcept>

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

/*! \param pdata Particle Data to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\post memory is allocated and all parameters lj1 and lj2 are set to 0.0
	\note The LJForceComputeGPU does not own the Neighborlist, the caller should
		delete the neighborlist when done.
*/
LJForceComputeGPU::LJForceComputeGPU(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut) 
	: LJForceCompute(pdata, nlist, r_cut), m_params_changed(true), m_ljparams(NULL), m_block_size(128)
	{
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// only one GPU is currently supported
	if (exec_conf.gpu.size() == 0)
		{
		cout << "Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl;
		throw std::runtime_error("Error initializing NeighborListNsqGPU");
		}
	if (exec_conf.gpu.size() != 1)
		{
		cout << "More than one GPU is not currently supported";
		throw std::runtime_error("Error initializing NeighborListNsqGPU");
		}	
	
	if (m_ntypes > MAX_NTYPES)
		{
		cerr << "LJForceComputeGPU cannot handle " << m_ntypes << " types" << endl;
		throw runtime_error("Error initializing LJForceComputeGPU");
		}

	// allocate the param data on the GPU and make sure it is up to date
	m_ljparams = gpu_alloc_ljparam_data();
	exec_conf.gpu[0]->call(bind(gpu_select_ljparam_data, m_ljparams, true));
	}
	

LJForceComputeGPU::~LJForceComputeGPU()
	{
	assert(m_ljparams);

	// deallocate our memory
	gpu_free_ljparam_data(m_ljparams);
	m_ljparams = NULL;
	}
	
/*! \param block_size Size of the block to run on the device
	Performance of the code may be dependant on the block size run
	on the GPU. \a block_size should be set to be a multiple of 32.
	\todo error check value
*/
void LJForceComputeGPU::setBlockSize(int block_size)
	{
	m_block_size = block_size;
	}

/*! \post The parameters \a lj1 and \a lj2 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
	\note \a lj? are low level parameters used in the calculation. In order to specify
	these for a normal lennard jones formula (with alpha), they should be set to the following.
	- \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
	- \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);
	
	Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
	is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
	for will have lj1 and lj2 set to 0.0.
	
	\param typ1 Specifies one type of the pair
	\param typ2 Specifies the second type of the pair
	\param lj1 First parameter used to calcluate forces
	\param lj2 Second parameter used to calculate forces
*/
void LJForceComputeGPU::setParams(unsigned int typ1, unsigned int typ2, Scalar lj1, Scalar lj2)
	{
	assert(m_ljparams);
	if (typ1 >= m_ntypes || typ2 >= m_ntypes)
		{
		cerr << "Trying to set LJ params for a non existant type! " << typ1 << "," << typ2 << endl;
		throw runtime_error("LJForceComputeGpu::setParams argument error");
		}
	
	// set coeffs in both symmetric positions in the matrix
	m_ljparams->coeffs[typ1*MAX_NTYPES + typ2] = make_float2(lj1, lj2);
	m_ljparams->coeffs[typ2*MAX_NTYPES + typ1] = make_float2(lj1, lj2);
	
	m_params_changed = true;
	}
		
/*! \post The lennard jones forces are computed for the given timestep on the GPU. 
	The neighborlist's compute method is called to ensure that it is up to date
	before forces are computed.
 	\param timestep Current time step of the simulation
*/
void LJForceComputeGPU::computeForces(unsigned int timestep)
	{
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();	
	
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile
	if (m_prof)
		m_prof->push("LJ.GPU");
	
	// The GPU implementation CANNOT handle a half neighborlist, error out now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	if (third_law)
		throw runtime_error("LJForceComputeGPU cannot handle a half neighborlist");
	
	// access the neighbor list, which just selects the neighborlist into the device's memory, copying
	// it there if needed
	gpu_nlist_array nlist = m_nlist->getListGPU();

	// access the particle data, like the neighborlist, this is just putting the pointers into the 
	// right constant memory and copying data to those pointers if needed
	gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	// tally up the number of forces calculated (but only if we are profiling)
	int64_t n_calc = 0;
	if (m_prof)
		{
		// keeping old profiling method around just in case
		/*m_prof->push("Profiling overhead");
		// access the neighbor list
    	const vector< vector< unsigned int > >& full_list = m_nlist->getList();
		// to tally up the number of computations made, we need to add consider that 
		// each warp calculates the same number of forces, even if the nlist isn't that long there
		const int warp_size = 1;
		for (unsigned int start = 0; start < full_list.size(); start += warp_size)
			{
			unsigned int max_n = 0;
			for (unsigned int i = start; (i < start+warp_size) && (i < full_list.size()); i++)
				{
				if (full_list[i].size() > max_n)
					max_n = full_list[i].size();
				}
			// tally up the number of computations performed
			n_calc += max_n * warp_size;
			}
		m_prof->pop();*/
		
		// efficiently estimate the number of calculations performed based on the 
		// neighbor list's esitmate of NNeigh
		Scalar avg_neigh = m_nlist->estimateNNeigh();
		n_calc = int64_t(avg_neigh * m_pdata->getN());
		}

	// actually perform the calculation
	if (m_params_changed)
		{
		exec_conf.gpu[0]->call(bind(gpu_select_ljparam_data, m_ljparams, m_params_changed));
		m_params_changed = false;
		}
	
	if (m_prof)
		m_prof->push("Compute");
	exec_conf.gpu[0]->call(bind(gpu_ljforce_sum, m_d_forces, &pdata, &box, &nlist, m_r_cut * m_r_cut, m_block_size));

	// FLOPS: 34 for each calc
	int64_t flops = (34+4)*n_calc;
		
	// memory transferred: 4*sizeof(Scalar) + sizeof(int) for each n_calc
	// PLUS an additional 8*sizeof(Scalar) + sizeof(int) for each particle
	int64_t mem_transfer = 0;
	mem_transfer += (4*sizeof(Scalar) + sizeof(int)) * n_calc;
	mem_transfer += (8*sizeof(Scalar) + sizeof(int)) * m_pdata->getN();
	
	m_pdata->release();
	
	// the force data is now only up to date on the gpu
	m_data_location = gpu;

	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop(flops, mem_transfer);
		m_prof->pop();
		}
	
	}

#ifdef USE_PYTHON
void export_LJForceComputeGPU()
	{
	class_<LJForceComputeGPU, boost::shared_ptr<LJForceComputeGPU>, bases<LJForceCompute>, boost::noncopyable >
		("LJForceComputeGPU", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<NeighborList>, Scalar >())
		.def("setBlockSize", &LJForceComputeGPU::setBlockSize)
		;
	}
#endif
