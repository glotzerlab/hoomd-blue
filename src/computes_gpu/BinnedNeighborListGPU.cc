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
// Maintainer: joaander

/*! \file BinnedNeighborListGPU.cc
	\brief Defines the BinnedNeighborListGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )

// windoze calls isnan by a different name....
#include <float.h>
#define isnan _isnan
#endif

#include "BinnedNeighborListGPU.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <math.h>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param pdata Particle data the neighborlist is to compute neighbors for
	\param r_cut Cuttoff radius under which particles are considered neighbors
	\param r_buff Buffer distance to include around the cutoff
	\post The neighbor list is initialized and the list memory has been allocated,
		but the list will not be computed until compute is called.
	\post The storage mode defaults to full
	\sa NeighborList
*/
BinnedNeighborListGPU::BinnedNeighborListGPU(boost::shared_ptr<ParticleData> pdata, Scalar r_cut, Scalar r_buff) : NeighborList(pdata, r_cut, r_buff), m_block_size(64)
	{
	// can't run on the GPU if there aren't any GPUs in the execution configuration
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a BinnedNeighborListGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing BinnedNeighborListGPU");
		}
	// setup the GPU data pointers
	m_gpu_bin_data.resize(exec_conf.gpu.size());
	
	m_storage_mode = full;
	// this is a bit of a trick, but initialize the last allocated Mx,My,Mz values to bogus settings so that
	// updatBins is sure to reallocate them
	// BUT, updateBins is going to free the arrays first, so allocate some dummy arrays
	allocateGPUBinData(1,1,1,1);

	// a reasonable default Nmax. This will expand as needed.
	m_Nmax = 128;
	m_curNmax = 0;
	m_avgNmax = Scalar(0.0);

	// ulf workaround setup
	#ifndef DISABLE_ULF_WORKAROUND
	cudaDeviceProp deviceProp;
	int dev;
	exec_conf.gpu[0]->call(bind(cudaGetDevice, &dev));
	exec_conf.gpu[0]->call(bind(cudaGetDeviceProperties, &deviceProp, dev));
	
	m_ulf_workaround = false;
	
	// the ULF workaround is needed on all compute 1.1 GPUs
	if (deviceProp.major == 1 && deviceProp.minor == 1)
		m_ulf_workaround = true;
		
	if (m_ulf_workaround)
		cout << "Notice: ULF bug workaround enabled for BinnedNeighborListGPU" << endl;
	#else
	m_ulf_workaround = false;
	#endif
	
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
	
/*! \param Mx Number of bins in the x direction
	\param My Number of bins in the y direction
	\param Mz Number of bins in the z direction
	\param Nmax Maximum number of particles stored in any given bin

	\pre Bins are not allocated
	\post All memory needed for the bin data structure on the GPU is allocated along with the host mirror versions
*/
void BinnedNeighborListGPU::allocateGPUBinData(unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
	{
	assert(exec_conf.gpu.size() >= 1);
	
	// use mallocPitch to make sure that memory accesses are coalesced	
	size_t pitch;

	// allocate and zero device memory
	if (Mx*My*Mz*Nmax >= 500000*128)
		cout << "***Warning! Allocating abnormally large cell list: " << Mx << " " << My << " " << Mz << " " << Nmax << endl;

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		// setup the dimensions
		m_gpu_bin_data[cur_gpu].Mx = Mx;
		m_gpu_bin_data[cur_gpu].My = My;
		m_gpu_bin_data[cur_gpu].Mz = Mz;
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocPitchHack, (void**)((void*)&m_gpu_bin_data[cur_gpu].idxlist), &pitch, Nmax*sizeof(unsigned int), Mx*My*Mz));
		// want pitch in elements, not bytes
		Nmax = (int)pitch / sizeof(unsigned int);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_bin_data[cur_gpu].idxlist, 0, pitch * Mx*My*Mz));

		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_bin_data[cur_gpu].bin_size), Mx*My*Mz*sizeof(unsigned int)));
	
		cudaChannelFormatDesc idxlist_desc = cudaCreateChannelDesc< unsigned int >();
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocArray, &m_gpu_bin_data[cur_gpu].idxlist_array, &idxlist_desc, Nmax, Mx*My*Mz));
	
		// allocate the bin adjacent list array
		cudaChannelFormatDesc bin_adj_desc = cudaCreateChannelDesc< int >();
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocArray, &m_gpu_bin_data[cur_gpu].bin_adj_array, &bin_adj_desc, Mx*My*Mz, 27));
		}
	
	// allocate and zero host memory
	exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_idxlist), pitch * Mx*My*Mz, cudaHostAllocPortable) );
	memset((void*)m_host_idxlist, 0, pitch*Mx*My*Mz);
	
	// allocate the host bin adj array
	int *bin_adj_host = new int[Mx*My*Mz*27];
	
	// initialize the coord and bin adj arrays
	for (int i = 0; i < (int)Mx; i++)
		{
		for (int j = 0; j < (int)My; j++)
			{
			for (int k = 0; k < (int)Mz; k++)
				{
				int bin = i*Mz*My + j*Mz + k;
				
				// loop over neighboring bins
				int cur_adj = 0;
				for (int neigh_i = i-1; neigh_i <= i+1; neigh_i++)
					{
					for (int neigh_j = j-1; neigh_j <= j+1; neigh_j++)
						{
						for (int neigh_k = k-1; neigh_k <= k+1; neigh_k++)
							{
							int a = neigh_i;
							if (a < 0) a+= Mx;
							if (a >= (int)Mx) a-= Mx;
							
							int b = neigh_j;
							if (b < 0) b+= My;
							if (b >= (int)My) b-= My;
							
							int c = neigh_k;
							if (c < 0) c+= Mz;
							if (c >= (int)Mz) c-= Mz;
							
							int neigh_bin = a*Mz*My + b*Mz + c;
							bin_adj_host[bin + cur_adj*Mx*My*Mz] = neigh_bin;
							cur_adj++;
							}
						}
					}
				}
			}
		}
	// sort to improve memory access pattern
	unsigned int nbins = Mx*My*Mz;
	for (unsigned int cur_bin = 0; cur_bin < nbins; cur_bin++)
		{
		bool swapped = false;
		do
			{
			swapped = false;
			for (unsigned int i = 0; i < 27-1; i++)
				{
				if (bin_adj_host[nbins*i + cur_bin] > bin_adj_host[nbins*(i+1) + cur_bin])
					{
					unsigned int tmp = bin_adj_host[nbins*(i+1) + cur_bin];
					bin_adj_host[nbins*(i+1) + cur_bin] = bin_adj_host[nbins*i + cur_bin];
					bin_adj_host[nbins*i + cur_bin] = tmp;
					swapped = true;
					}
				}
			} while (swapped);
		}
	
	// copy it to the device. This only needs to be done once
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);	
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpyToArray, m_gpu_bin_data[cur_gpu].bin_adj_array, 0, 0, bin_adj_host, 27*Mx*My*Mz*sizeof(int), cudaMemcpyHostToDevice));
		}
	
	// don't need the temporary bin adj data any more
	delete[] bin_adj_host;
	
	// allocate the coord_idxlist data
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		size_t pitch_coord;
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocPitchHack, (void**)((void*)&m_gpu_bin_data[cur_gpu].coord_idxlist), &pitch_coord, Mx*My*Mz*sizeof(float4), Nmax));
		// want width in elements, not bytes
		m_gpu_bin_data[cur_gpu].coord_idxlist_width = (int)pitch_coord / sizeof(float4);
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_bin_data[cur_gpu].coord_idxlist, 0, pitch_coord * Nmax));
		
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		cudaChannelFormatDesc coord_idxlist_desc = cudaCreateChannelDesc< float4 >();
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocArray, &m_gpu_bin_data[cur_gpu].coord_idxlist_array, &coord_idxlist_desc, m_gpu_bin_data[cur_gpu].coord_idxlist_width, Nmax));
		
		// assign allocated pitch
		m_gpu_bin_data[cur_gpu].Nmax = Nmax;
		}
	}
	
/*! \pre allocateGPUBinData() has been called
	\post Bin data is freed
*/
void BinnedNeighborListGPU::freeGPUBinData()
	{
	assert(exec_conf.gpu.size() >= 1);
	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{	
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		
		// free the device memory
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_bin_data[cur_gpu].idxlist));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFreeArray, m_gpu_bin_data[cur_gpu].idxlist_array));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_bin_data[cur_gpu].coord_idxlist));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_bin_data[cur_gpu].bin_size));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFreeArray, m_gpu_bin_data[cur_gpu].coord_idxlist_array));
		exec_conf.gpu[cur_gpu]->call(bind(cudaFreeArray, m_gpu_bin_data[cur_gpu].bin_adj_array));
	
		// set pointers to NULL so no one will think they are valid 
		m_gpu_bin_data[cur_gpu].idxlist = NULL;
		}
	
	// free the hsot memory
	exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_idxlist));
	m_host_idxlist = NULL;
	}

/*! Makes all the calls needed to bring the neighbor list up to date on the GPU.
	This requires building the cell list, copying it to the GPU and then 
	attempting to build the list repeatedly, increasing the allocated 
	memory each time until the list does not overflow.
*/
void BinnedNeighborListGPU::buildNlist()
	{
	assert(exec_conf.gpu.size() >= 1);
	
	// bin the particles
	updateBinsUnsorted();
		
	// copy those bins to the GPU
	if (m_prof) m_prof->push(exec_conf, "Bin copy");
		
	unsigned int nbytes = m_gpu_bin_data[0].Mx * m_gpu_bin_data[0].My * m_gpu_bin_data[0].Mz * m_gpu_bin_data[0].Nmax * sizeof(unsigned int);

	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpyToArray, m_gpu_bin_data[cur_gpu].idxlist_array, 0, 0, m_host_idxlist, nbytes, cudaMemcpyHostToDevice));
		exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpy, m_gpu_bin_data[cur_gpu].bin_size, &m_bin_sizes[0], sizeof(unsigned int)*m_gpu_bin_data[0].Mx*m_gpu_bin_data[0].My*m_gpu_bin_data[0].Mz, cudaMemcpyHostToDevice));
		}
	exec_conf.syncAll();
	
	if (m_prof) m_prof->pop(exec_conf, 0, nbytes*(unsigned int)exec_conf.gpu.size());
	// transpose the bins for a better memory access pattern on the GPU
	
	if (m_prof) m_prof->push(exec_conf, "Transpose");
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_nlist_idxlist2coord, &pdata[cur_gpu], &m_gpu_bin_data[cur_gpu], m_curNmax, 256));
		}
	exec_conf.syncAll();
	
	m_pdata->release();
		
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{	
		exec_conf.gpu[cur_gpu]->callAsync(bind(cudaMemcpyToArray, m_gpu_bin_data[cur_gpu].coord_idxlist_array, 0, 0, m_gpu_bin_data[cur_gpu].coord_idxlist, m_gpu_bin_data[cur_gpu].coord_idxlist_width*m_curNmax*sizeof(float4), cudaMemcpyDeviceToDevice));
		}
	if (m_prof) m_prof->pop(exec_conf, 0);
		
	// update the neighbor list using the bins. Need to check for overflows
	// and increase the size of the list as needed
	updateListFromBins();
		
	int overflow = 0;
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		int overflow_tmp = 0;
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &overflow_tmp, m_gpu_nlist[cur_gpu].overflow, sizeof(int), cudaMemcpyDeviceToHost));
		overflow = overflow || overflow_tmp;
		}
		
	while (overflow)
		{
		int new_height = (int)(Scalar(m_gpu_nlist[0].height) * 1.2);
		// cout << "Notice: Neighborlist overflowed on GPU, expanding to " << new_height << " neighbors per particle..." << endl;
		freeGPUData();
		allocateGPUData(new_height);
		updateExclusionData();
		
		updateListFromBins();
		overflow = 0;
		for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
			{
			int overflow_tmp = 0;
			exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
			exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, &overflow_tmp, m_gpu_nlist[cur_gpu].overflow, sizeof(int), cudaMemcpyDeviceToHost));
			overflow = overflow || overflow_tmp;
			}
		}
		
	m_data_location = gpu;
	}

/*! Particles are copied to the CPU and the binning step is done, putting the index of each particle
	into each bin. This is all done in the host memory copies of the bin arrays, they are not
	copied to the GPU here.

	\pre GPU Bin data has been allocated.
*/
void BinnedNeighborListGPU::updateBinsUnsorted()
	{
	assert(m_pdata);
		
	// start up the profile
	if (m_prof) m_prof->push(exec_conf, "Bin");

	// acquire the particle data
	ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
	
	// calculate the bin dimensions
	const BoxDim& box = m_pdata->getBox();
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	
	m_Mx = int((box.xhi - box.xlo) / (m_r_cut + m_r_buff));
	m_My = int((box.yhi - box.ylo) / (m_r_cut + m_r_buff));
	m_Mz = int((box.zhi - box.zlo) / (m_r_cut + m_r_buff));
	if (m_Mx < 3 || m_My < 3 || m_Mz < 3)
		{
		cerr << endl << "***Error! BinnedNeighborListGPU doesn't work on boxes where r_cut+r_buff is greater than 1/3 any box dimension" << endl << endl;
		throw runtime_error("Error updating neighborlist bins");
		}

	// TODO: this should really be determined as a minimum of the memcpy pitch and the 2D texture dimensions
	// decrease the number of bins if it exceeds 16384
	if (m_Mx * m_My *m_Mz > 16384)
		{
		float scale_factor = powf(16384.0f / float(m_Mx * m_My *m_Mz), 1.0f/3.0f);
		m_Mx = int(float(m_Mx)*scale_factor);
		m_My = int(float(m_My)*scale_factor);
		m_Mz = int(float(m_Mz)*scale_factor);
		}
	
	// if these dimensions are different than the previous dimensions, reallocate
	if (m_Mx != m_last_Mx || m_My != m_last_My || m_Mz != m_last_Mz)
		{
		freeGPUBinData();
		allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data[0].Nmax;
		
		m_last_Mx = m_Mx;
		m_last_My = m_My;
		m_last_Mz = m_Mz;
		m_bin_sizes.resize(m_Mx*m_My*m_Mz);
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
		if (isnan(arrays.x[n]) || isnan(arrays.y[n]) || isnan(arrays.z[n]))
			{
			cerr << endl << "***Error! Particle " << n << " has NaN for its position." << endl << endl;
			throw runtime_error("Error binning particles");
			}
		
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
		// check if the particle is inside
		if (bin >= m_Mx*m_My*m_Mz)
			{
			cerr << endl << "***Error! Elvis has left the building (particle " << n << " is no longer in the simulation box)." << endl << endl;
			throw runtime_error("Error binning particles");
			}
		
		unsigned int size = m_bin_sizes[bin];
	
		// track the size of the largest bin
		if (size+1 > m_curNmax)
			m_curNmax = size+1;

		// make sure we don't overflow
		if (size < m_Nmax)
			{
			m_host_idxlist[size + bin*m_Nmax] = n;
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
	if (m_prof) m_prof->pop(exec_conf);

	// we aren't done yet, if there was an overflow, update m_Nmax and recurse to make sure the list is fully up to date
	// since we are now certain that m_Nmax will hold all of the particles, the recursion should only happen once
	if (overflow)
		{
		// reallocate memory first, so there is room
		m_Nmax = overflow_value+1;
		freeGPUBinData();
		allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data[0].Nmax;
		updateBinsUnsorted();
		}
	// note, we don't copy the binned values to the device yet, that is for the compute to do
	}

/*! Sets up and executes the needed kernel calls to generate the actual neighbor list
	from the previously binned data.
	
	\pre updateBinsUnsorted() has been called.
	\pre The bin data has been copied to the GPU and transposed into the idxlist_coord array
	
	Calls gpu_compute_nlist_binned to do the dirty work.
*/
void BinnedNeighborListGPU::updateListFromBins()
	{
	assert(exec_conf.gpu.size() >= 1);
	
	if (m_storage_mode != full)
		{
		cerr << endl << "***Error! Only full mode nlists can be generated on the GPU" << endl << endl;
		throw runtime_error("Error computing neighbor list");
		}

	// sanity check
	assert(m_pdata);
		
	// start up the profile
	if (m_prof) m_prof->push(exec_conf, "Build list");
		
	// access the particle data
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	Scalar r_max_sq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{	
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_nlist_binned, m_gpu_nlist[cur_gpu], pdata[cur_gpu], box, m_gpu_bin_data[cur_gpu], r_max_sq, m_curNmax, m_block_size, m_ulf_workaround));
		}
		
	exec_conf.syncAll();
	
	m_pdata->release();

	int64_t flops = int64_t(m_pdata->getN() * (9 + 27 * m_avgNmax * (15 + 5 + 1)));
	int64_t mem_transfer = int64_t(m_pdata->getN() * (32 + 4 + 8 + 27 * (4 + m_avgNmax * 16) + estimateNNeigh() * 4));
	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	}

/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since the last call
		to this method that returned true.
	\returns false If none of the particles has been moved more than 1/2 of the buffer distance since the last call to this
		method that returned true.
*/
bool BinnedNeighborListGPU::distanceCheck()
	{
	assert(exec_conf.gpu.size() >= 1);
	
	// scan through the particle data arrays and calculate distances
	if (m_prof) m_prof->push(exec_conf, "Dist check");
		
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	// create a temporary copy of r_max sqaured
	Scalar r_buffsq = (m_r_buff/Scalar(2.0)) * (m_r_buff/Scalar(2.0));
	
	int result = 0;
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		int result_tmp;
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(gpu_nlist_needs_update_check, &pdata[cur_gpu], &box, &m_gpu_nlist[cur_gpu], r_buffsq, &result_tmp));
		result = result | result_tmp;
		}

	m_pdata->release();

	if (m_prof) m_prof->pop(exec_conf);

	return result;
	}

/*! Does nothing in BinnedNeighborListGPU. The last position values are set whenever the neighbor list is built.
*/
void BinnedNeighborListGPU::setLastUpdatedPos()
	{
	}

/*! This method just adds a few statistic numbers specific to this class. The base class method is called
	first to get the more general stats from there.
*/
void BinnedNeighborListGPU::printStats()
	{
	NeighborList::printStats();

	cout << "Nmax = " << m_Nmax << " / curNmax = " << m_curNmax << endl;
	int Nbins = m_gpu_bin_data[0].Mx * m_gpu_bin_data[0].My * m_gpu_bin_data[0].Mz;
	cout << "bins Nmax = " << m_gpu_bin_data[0].Nmax << " / Nbins = " << Nbins << endl;
	}

void export_BinnedNeighborListGPU()
	{
	class_<BinnedNeighborListGPU, boost::shared_ptr<BinnedNeighborListGPU>, bases<NeighborList>, boost::noncopyable >
		("BinnedNeighborListGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		.def("setBlockSize", &BinnedNeighborListGPU::setBlockSize)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
