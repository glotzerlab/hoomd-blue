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

/*! \file BinnedNeighborListGPU.cc
	\brief Defines the BinnedNeighborListGPU class
*/

#include "BinnedNeighborListGPU.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

/*! \param pdata Particle data the neighborlist is to compute neighbors for
	\param r_cut Cuttoff radius under which particles are considered neighbors
	\param r_buff Buffer distance to include around the cutoff
	\post NeighborList is initialized and the list memory has been allocated,
		but the list will not be computed until compute is called.
	\post The storage mode defaults to full and cannot be set to half
	\sa NeighborList
*/
BinnedNeighborListGPU::BinnedNeighborListGPU(boost::shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff) : NeighborList(pdata, r_cut, r_buff)
	{
	// check the execution configuration
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	// only one GPU is currently supported
	if (exec_conf.gpu.size() == 0)
		{
		cout << "Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl;
		throw std::runtime_error("Error initializing BinnedNeighborListGPU");
		}
	if (exec_conf.gpu.size() != 1)
		{
		cout << "More than one GPU is not currently supported";
		throw std::runtime_error("Error initializing BinnedNeighborListGPU");
		}
	
	m_storage_mode = full;
	// this is a bit of a trick, but initialize the last allocated Mx,My,Mz values to bogus settings so that
	// updatBins is sure to reallocate them
	// BUT, updateBins is going to free the arrays first, so allocate some dummy arrays
	allocateGPUBinData(1,1,1,1);

	// a reasonable default Nmax. This will expand as needed.
	m_Nmax = 128;
	m_curNmax = 0;
	m_avgNmax = Scalar(0.0);

	// default block size is the highest performance in testing
	m_block_size = 256;

	// bogus values for last value
	m_last_Mx = INT_MAX;
	m_last_My = INT_MAX;
	m_last_Mz = INT_MAX;
	m_Mx = 0;
	m_My = 0;
	m_Mz = 0;
	}

BinnedNeighborListGPU::~BinnedNeighborListGPU()
	{
	freeGPUBinData();
	}
	
void BinnedNeighborListGPU::allocateGPUBinData(unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	assert(exec_conf.gpu.size() == 1);
	
	// setup the dimensions
	m_gpu_bin_data.Mx = Mx;
	m_gpu_bin_data.My = My;
	m_gpu_bin_data.Mz = Mz;

	// use mallocPitch to make sure that memory accesses are coalesced	
	size_t pitch;

	// allocate and zero device memory
	if (Mx*My*Mz*Nmax >= 500000*128)
		cout << "Allocating abnormally large cell list: " << Mx << " " << My << " " << Mz << " " << Nmax << endl;

	exec_conf.gpu[0]->call(bind(cudaMallocPitch, (void**)((void*)&m_gpu_bin_data.idxlist), &pitch, Nmax*sizeof(unsigned int), Mx*My*Mz));
	// want pitch in elements, not bytes
	Nmax = (int)pitch / sizeof(unsigned int);
	exec_conf.gpu[0]->call(bind(cudaMemset, (void*) m_gpu_bin_data.idxlist, 0, pitch * Mx*My*Mz));
	cudaChannelFormatDesc idxlist_desc = cudaCreateChannelDesc< int >();
	exec_conf.gpu[0]->call(bind(cudaMallocArray, &m_gpu_bin_data.idxlist_array, &idxlist_desc, Nmax, Mx*My*Mz));
	
	// allocate the bin coord array
	exec_conf.gpu[0]->call(bind(cudaMalloc, (void**)((void*)&m_gpu_bin_data.bin_coord), Mx*My*Mz*sizeof(uint4)));
	
	// allocate and zero host memory
	exec_conf.gpu[0]->call(bind(cudaMallocHost, (void**)((void*)&m_host_idxlist), pitch * Mx*My*Mz) );
	memset((void*)m_host_idxlist, 0, pitch*Mx*My*Mz);
	
	// allocate the bin coord array
	m_host_bin_coord = (uint4*)malloc(sizeof(uint4) * Mx*My*Mz);
	// initialize the coord array
	for (unsigned int i = 0; i < Mx; i++)
		{
		for (unsigned int j = 0; j < My; j++)
			{
			for (unsigned int k = 0; k < Mz; k++)
				{
				int bin = i*Mz*My + j*Mz + k;
				m_host_bin_coord[bin].x = i;
				m_host_bin_coord[bin].y = j;
				m_host_bin_coord[bin].z = k;
				m_host_bin_coord[bin].w = 0;
				}
			}
		}
	// copy it to the device. This only needs to be done once
	exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_bin_data.bin_coord, m_host_bin_coord, sizeof(uint4)*Mx*My*Mz, cudaMemcpyHostToDevice));
	
	// assign allocated pitch
	m_gpu_bin_data.Nmax = Nmax;
	}
	

void BinnedNeighborListGPU::freeGPUBinData()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	assert(exec_conf.gpu.size() == 1);
	
	// free the device memory
	exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_bin_data.idxlist));
	exec_conf.gpu[0]->call(bind(cudaFreeArray, m_gpu_bin_data.idxlist_array));
	exec_conf.gpu[0]->call(bind(cudaFree, m_gpu_bin_data.bin_coord));
	// free the hsot memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_idxlist));
	free(m_host_bin_coord);

	// set pointers to NULL so no one will think they are valid 
	m_gpu_bin_data.idxlist = NULL;
	m_host_idxlist = NULL;
	}

/*! Updates the neighborlist if it has not yet been updated this times step
 	\param timestep Current timestep to compute for
*/
void BinnedNeighborListGPU::compute(unsigned int timestep)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	assert(exec_conf.gpu.size() == 1);
	
	// skip if we shouldn't compute this step
	if (!shouldCompute(timestep) && !m_force_update)
		return;

	if (m_storage_mode != full)
		{
		cerr << "Only full mode nlists can be generated on the GPU" << endl;
		exit(1);
		}

	if (m_prof)
		m_prof->push("Nlist.GPU");
	
	// need to update the exclusion data if anything has changed
	if (m_force_update)
		updateExclusionData();

	// update the list (if it needs it)
	if (needsUpdating(timestep))
		{
		updateBinsUnsorted();
		
		if (m_prof)
			m_prof->push("Bin copy");
		
		unsigned int nbytes = m_gpu_bin_data.Mx * m_gpu_bin_data.My * m_gpu_bin_data.Mz * m_gpu_bin_data.Nmax * sizeof(unsigned int);

		exec_conf.gpu[0]->call(bind(cudaMemcpy, m_gpu_bin_data.idxlist, m_host_idxlist,
			nbytes, cudaMemcpyHostToDevice));
		exec_conf.gpu[0]->call(bind(cudaMemcpyToArray, m_gpu_bin_data.idxlist_array, 0, 0, m_host_idxlist, nbytes,
			cudaMemcpyHostToDevice));
		
		if (m_prof)
			{
			int nbytes = m_gpu_bin_data.Mx * m_gpu_bin_data.My *
						m_gpu_bin_data.Mz * m_gpu_bin_data.Nmax *
						sizeof(unsigned int);
						
			m_prof->pop(0, nbytes);
			}
		
		// update the neighbor list using the bins. Need to check for overflows
		// and increase the size of the list as needed
		updateListFromBins();
		
		int overflow = 0;
		exec_conf.gpu[0]->call(bind(cudaMemcpy, &overflow, m_gpu_nlist.overflow, sizeof(int), cudaMemcpyDeviceToHost));
		while (overflow)
			{
			int new_height = m_gpu_nlist.height * 2;
			cout << "Neighborlist overflowed on GPU, expanding to " << new_height << " neighbors per particle..." << endl;
			freeGPUData();
			allocateGPUData(new_height);
			updateExclusionData();
			
			updateListFromBins();
			exec_conf.gpu[0]->call(bind(cudaMemcpy, &overflow, m_gpu_nlist.overflow, sizeof(int), cudaMemcpyDeviceToHost));
			}
			
		#ifdef USE_CUDA
		// after computing, the device now resides on the CPU
		m_data_location = gpu;
		#endif
		}
		
	if (m_prof)	m_prof->pop();
	}

void BinnedNeighborListGPU::updateBinsUnsorted()
	{
	assert(m_pdata);

	// start up the profile
	if (m_prof)
		m_prof->push("Bin");

	// acquire the particle data
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	
	// calculate the bin dimensions
	const BoxDim& box = m_pdata->getBox();
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	m_Mx = int((box.xhi - box.xlo) / (m_r_cut + m_r_buff));
	m_My = int((box.yhi - box.ylo) / (m_r_cut + m_r_buff));
	m_Mz = int((box.zhi - box.zlo) / (m_r_cut + m_r_buff));
	if (m_Mx == 0)
		m_Mx = 1;
	if (m_My == 0)
		m_My = 1;
	if (m_Mz == 0)
		m_Mz = 1;

	// if these dimensions are different than the previous dimensions, reallocate
	if (m_Mx != m_last_Mx || m_My != m_last_My || m_Mz != m_last_Mz)
		{
		freeGPUBinData();
		allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.Nmax;
		
		m_last_Mx = m_Mx;
		m_last_My = m_My;
		m_last_Mz = m_Mz;
		}

	// make even bin dimensions
	Scalar binx = (box.xhi - box.xlo) / Scalar(m_Mx);
	Scalar biny = (box.yhi - box.ylo) / Scalar(m_My);
	Scalar binz = (box.zhi - box.zlo) / Scalar(m_Mz);

	// precompute scale factors to eliminate division in inner loop
	Scalar scalex = Scalar(1.0) / binx;
	Scalar scaley = Scalar(1.0) / biny;
	Scalar scalez = Scalar(1.0) / binz;

	// setup the memory arrays
	m_bin_sizes.resize(m_Mx*m_My*m_Mz);
	for (unsigned int i = 0; i < m_Mx*m_My*m_Mz; i++)
		m_bin_sizes[i] = 0;
	
	// clear the bins to 0xffffffff which means no particle in that bin
	memset((void*)m_host_idxlist, 0xff, sizeof(unsigned int)*m_Mx*m_My*m_Mz*m_Nmax);

	// reset the counter that keeps track of the current size of the largest bin
	m_curNmax = 0;
	
	// for each particle
	bool overflow = false;
	unsigned int overflow_value = 0;
	for (unsigned int n = 0; n < arrays.nparticles; n++)
		{
		// find the bin each particle belongs in
		unsigned int ib = (unsigned int)((arrays.x[n]-box.xlo)*scalex);
		unsigned int jb = (unsigned int)((arrays.y[n]-box.ylo)*scaley);
		unsigned int kb = (unsigned int)((arrays.z[n]-box.zlo)*scalez);

		// need to handle the case where the particle is exactly at the box hi
		if (ib == m_Mx)
			ib = 0;
		if (jb == m_My)
			jb = 0;
		if (kb == m_Mz)
			kb = 0;

		// sanity check
		assert(ib >= 0 && ib < m_Mx && jb >= 0 && jb < m_My && kb >= 0 && kb < m_Mz);

		// record its bin
		unsigned int bin = ib*(m_Mz*m_My) + jb * m_Mz + kb;
		unsigned int size = m_bin_sizes[bin];
	
		// track the size of the largest bin
		if (size+1 > m_curNmax)
			m_curNmax = size+1;

		// make sure we don't overflow
		if (size < m_Nmax)
			{
			m_host_idxlist[bin*m_Nmax + size] = n;
			}
		else
			{
			overflow = true;
			if (size > overflow_value)
				overflow_value = size;
			}
		m_bin_sizes[bin]++;
		}
		
	m_avgNmax = Scalar(0.0);
	for (unsigned int i = 0; i < m_Mx * m_My * m_Mz; i++)
		{
		m_avgNmax += m_bin_sizes[i];
		}
	m_avgNmax /= Scalar(m_Mx * m_My * m_Mz);
	
	m_pdata->release();

	// update profile
	if (m_prof)
		m_prof->pop(6*arrays.nparticles, (3*sizeof(Scalar) + (3)*sizeof(unsigned int))*arrays.nparticles);

	// we aren't done yet, if there was an overflow, update m_Nmax and recurse to make sure the list is fully up to date
	// since we are now certain that m_Nmax will hold all of the particles, the recursion should only happen once
	if (overflow)
		{
		// reallocate memory first, so there is room
		m_Nmax = overflow_value+1;
		freeGPUBinData();
		allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.Nmax;
		updateBinsUnsorted();
		}
	// note, we don't copy the binned values to the device yet, that is for the compute to do
	}




/*! 
*/
void BinnedNeighborListGPU::updateListFromBins()
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	assert(exec_conf.gpu.size() == 1);
	
	// sanity check
	assert(m_pdata);
		
	// start up the profile
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->push("Build list");
		}
		
	// access the particle data
	gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU(); 
	
	Scalar r_max_sq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
	
	exec_conf.gpu[0]->call(bind(gpu_nlist_binned, &pdata, &box, &m_gpu_bin_data, &m_gpu_nlist, r_max_sq, m_curNmax, m_block_size));
	
	m_pdata->release();

	// upate the profile
	int nbins = m_gpu_bin_data.Mx * m_gpu_bin_data.My * m_gpu_bin_data.Mz;
	uint64_t nthreads = nbins * m_curNmax;
	
	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		// each thread computes 21 flops for each comparison. There are 27*m_avgNmax comparisons per thread.
		// each thread reads 592 bytes: this includes reading its particle position, exclusion,
		// cell lists for neighboring bins and those particle's positions.
		m_prof->pop(int64_t(m_pdata->getN() * 27 * m_avgNmax * 21), nthreads * 592);
		}
	}
	


//! Test if the list needs updating
bool BinnedNeighborListGPU::needsUpdating(unsigned int timestep)
	{
	const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
	assert(exec_conf.gpu.size() == 1);	
	
	if (timestep < (m_last_updated_tstep + m_every) && !m_force_update)
		return false;
	
	if (m_force_update)
		{
		m_force_update = false;
		m_forced_updates += m_pdata->getN();
		m_last_updated_tstep = timestep;
		return true;
		}
		
	if (m_r_buff < 1e-6)
		return true;

	// scan through the particle data arrays and calculate distances
	if (m_prof)
		m_prof->push("Dist check");
		
	gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	// create a temporary copy of r_max sqaured
	Scalar r_buffsq = (m_r_buff/Scalar(2.0)) * (m_r_buff/Scalar(2.0));	
	
	int result = 0;
	exec_conf.gpu[0]->call(bind(gpu_nlist_needs_update_check, &pdata, &box, &m_gpu_nlist, r_buffsq, &result));
	

	m_pdata->release();

	if (m_prof)
		{
		exec_conf.gpu[0]->call(bind(cudaThreadSynchronize));
		m_prof->pop();
		}

	if (result)
		{
		m_last_updated_tstep = timestep;
		m_updates += m_pdata->getN();
		return true;
		}
	else
		return false;
	}

void BinnedNeighborListGPU::printStats()
	{
	NeighborList::printStats();

	cout << "Nmax = " << m_Nmax << " / curNmax = " << m_curNmax << endl;
	int Nbins = m_gpu_bin_data.Mx * m_gpu_bin_data.My * m_gpu_bin_data.Mz;
	cout << "bins Nmax = " << m_gpu_bin_data.Nmax << " / Nbins = " << Nbins << endl;
	}

#ifdef USE_PYTHON
void export_BinnedNeighborListGPU()
	{
	class_<BinnedNeighborListGPU, boost::shared_ptr<BinnedNeighborListGPU>, bases<NeighborList>, boost::noncopyable >
		("BinnedNeighborListGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		.def("setBlockSize", &BinnedNeighborListGPU::setBlockSize)
		;
	}	
#endif
