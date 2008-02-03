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

#include "BinnedNeighborListGPU.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

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
	m_storage_mode = full;
	// this is a bit of a trick, but initialize the last allocated Mx,My,Mz values to bogus settings so that
	// updatBins is sure to reallocate them
	// BUT, updateBins is going to free the arrays first, so allocate some dummy arrays
	gpu_alloc_bin_data(&m_gpu_bin_data, 1,1,1,1);

	// a reasonable default Nmax. (one warp) This will expand as needed.
	m_Nmax = 128;
	m_curNmax = 0;
	m_avgNmax = Scalar(0.0);

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
	gpu_free_bin_data(&m_gpu_bin_data);
	}

/*! Updates the neighborlist if it has not yet been updated this times step
 	\param timestep Current timestep to compute for
*/
void BinnedNeighborListGPU::compute(unsigned int timestep)
	{
	checkForceUpdate();
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
		
		gpu_copy_bin_data_htod(&m_gpu_bin_data);
		
		if (m_prof)
			{
			int nbytes = m_gpu_bin_data.h_array.Mx * m_gpu_bin_data.h_array.My * 
						m_gpu_bin_data.h_array.Mz * m_gpu_bin_data.h_array.Nmax * 
						sizeof(unsigned int);
						
			m_prof->pop(0, nbytes);
			}
			
		updateListFromBins();

		#ifdef USE_CUDA
		// after computing, the device now resides on the CPU
		m_data_location = gpu;
		#endif
		}
		
	if (m_prof)	m_prof->pop();
	}

/*! \todo document me
	\todo detect massive overflows
*/
/*void BinnedNeighborListGPU::updateBinsSorted()
	{
	assert(m_pdata);

	// start up the profile
	if (m_prof)
		m_prof->push("Update bins");

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
		gpu_free_bin_data(&m_gpu_bin_data);
		gpu_alloc_bin_data(&m_gpu_bin_data, m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.h_array.Nmax;
		
		m_last_Mx = m_Mx;
		m_last_My = m_My;
		m_last_Mz = m_Mz;
		}

	// make even bin dimensions
	Scalar binx = (box.xhi - box.xlo) / Scalar(m_Mx);
	Scalar biny = (box.yhi - box.ylo) / Scalar(m_Mx);
	Scalar binz = (box.zhi - box.zlo) / Scalar(m_Mx);

	// precompute scale factors to eliminate division in inner loop
	Scalar scalex = Scalar(1.0) / binx;
	Scalar scaley = Scalar(1.0) / biny;
	Scalar scalez = Scalar(1.0) / binz;

	// setup the memory arrays
	m_bin_sizes.resize(m_Mx*m_My*m_Mz);
	for (int i = 0; i < m_Mx*m_My*m_Mz; i++)
		m_bin_sizes[i] = 0;
	
	m_full_bin_sizes.resize(m_Mx*m_My*m_Mz);
	for (int i = 0; i < m_Mx*m_My*m_Mz; i++)
		m_full_bin_sizes[i] = 0;

	// clear the bins to 0xffffffff which means no particle in that bin
	memset((void*)m_gpu_bin_data.h_array.idxlist, 0xff, sizeof(unsigned int)*m_Mx*m_My*m_Mz*m_Nmax);
	memset((void*)m_gpu_bin_data.h_array.idxlist_full, 0xff, sizeof(unsigned int)*m_Mx*m_My*m_Mz*m_Nmax*27);

	// for each particle
	bool overflow = false;
	unsigned int overflow_value = 0;
	for (unsigned int n = 0; n < arrays.nparticles; n++)
		{
		// find the bin each particle belongs in
		int ib = int((arrays.x[n]-box.xlo)*scalex);
		int jb = int((arrays.y[n]-box.ylo)*scaley);
		int kb = int((arrays.z[n]-box.zlo)*scalez);

		// need to handle the case where the particle is exactly at the box hi
		if (ib == m_Mx)
			ib = 0;
		if (jb == m_My)
			jb = 0;
		if (kb == m_Mz)
			kb = 0;

		// sanity check
		assert(ib >= 0 && ib < int(m_Mx) && jb >= 0 && jb < int(m_My) && kb >= 0 && kb < int(m_Mz));

		// record its bin
		unsigned int bin = ib*(m_My*m_Mz) + jb * m_Mz + kb;
		
		int size = m_bin_sizes[bin];
		// make sure we don't overflow
		if (size < m_Nmax)
			{
			m_gpu_bin_data.h_array.idxlist[bin*m_Nmax + size] = n;
			}
		else
			{
			overflow = true;
			if (size > overflow_value)
				overflow_value = size;
			}
		m_bin_sizes[bin]++;

		for (int neigh_i = ib - 1; neigh_i <= ib + 1; neigh_i++)
			{
			for (int neigh_j = jb - 1; neigh_j <= jb + 1; neigh_j++)
				{
				for (int neigh_k = kb - 1; neigh_k <= kb + 1; neigh_k++)
					{
					int a = neigh_i;
					if (a <  0)
						a += m_Mx;
					if (a >= m_Mx)
						a -= m_Mx;
					
					int b = neigh_j;
					if (b <  0)
						b += m_My;
					if (b >= m_My)
						b -= m_My;
					
					int c = neigh_k;
					if (c <  0)
						c += m_Mz;
					if (c >= m_Mz)
						c -= m_Mz;

					int neigh_bin = a*(m_My*m_Mz) + b * m_Mz + c;
		
					int size = m_full_bin_sizes[neigh_bin];
					// make sure we don't overflow
					if (size < m_Nmax*27)
						{
						m_gpu_bin_data.h_array.idxlist_full[neigh_bin*m_Nmax*27 + size] = n;
						}
					m_full_bin_sizes[neigh_bin]++;
					}
				}
			}
		}

	m_pdata->release();

	// update profile
	if (m_prof)
		m_prof->pop(6*arrays.nparticles, (3*sizeof(Scalar) + (3+3*27)*sizeof(unsigned int))*arrays.nparticles);


	// we aren't done yet, if there was an overflow, update m_Nmax and recurse to make sure the list is fully up to date
	// since we are now certain that m_Nmax will hold all of the particles, the recursion should only happen once
	if (overflow)
		{
		// reallocate memory first, so there is room
		m_Nmax = overflow_value+1;
		gpu_free_bin_data(&m_gpu_bin_data);
		gpu_alloc_bin_data(&m_gpu_bin_data, m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.h_array.Nmax;
		updateBinsSorted();
		}
	// note, we don't copy the binned values to the device yet, that is for the compute to do
	}*/


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
		gpu_free_bin_data(&m_gpu_bin_data);
		gpu_alloc_bin_data(&m_gpu_bin_data, m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.h_array.Nmax;
		
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
	memset((void*)m_gpu_bin_data.h_array.idxlist, 0xff, sizeof(unsigned int)*m_Mx*m_My*m_Mz*m_Nmax);

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
		if (!(ib >= 0 && ib < int(m_Mx) && jb >= 0 && jb < int(m_My) && kb >= 0 && kb < int(m_Mz)))
			assert(ib >= 0 && ib < int(m_Mx) && jb >= 0 && jb < int(m_My) && kb >= 0 && kb < int(m_Mz));

		// record its bin
		unsigned int bin = ib*(m_Mz*m_My) + jb * m_Mz + kb;
		unsigned int size = m_bin_sizes[bin];
	
		// track the size of the largest bin
		if (size+1 > m_curNmax)
			m_curNmax = size+1;

		// make sure we don't overflow
		if (size < m_Nmax)
			{
			m_gpu_bin_data.h_array.idxlist[bin*m_Nmax + size] = n;
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
		gpu_free_bin_data(&m_gpu_bin_data);
		gpu_alloc_bin_data(&m_gpu_bin_data, m_Mx, m_My, m_Mz, m_Nmax);
		// reassign m_Nmax since it may have been expanded by the allocation process
		m_Nmax = m_gpu_bin_data.h_array.Nmax;
		updateBinsUnsorted();
		}

	// note, we don't copy the binned values to the device yet, that is for the compute to do
	}




/*! \todo document me
	\todo profile calculation
*/
void BinnedNeighborListGPU::updateListFromBins()
	{
	// sanity check
	assert(m_pdata);
		
	// start up the profile
	if (m_prof)
		{
		cudaThreadSynchronize();
		m_prof->push("Build list");
		}
		
	// access the particle data
	gpu_pdata_arrays pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU(); 
	
	// create a temporary copy of r_max sqaured
	int block_size = m_curNmax;
	if ((block_size & 31) != 0)
		block_size += 32 - (block_size & 31);

	#ifdef USE_CUDA_BUG_WORKAROUND
	block_size = m_Nmax;
	#endif
	
	Scalar r_max_sq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
	
	gpu_nlist_binned(&pdata, &box, &m_gpu_bin_data, &m_gpu_nlist, r_max_sq, block_size);
	//gpu_nlist_binned(&pdata, &box, &m_gpu_bin_data, &m_gpu_nlist, r_max_sq, m_Nmax);
	
	m_pdata->release();

	// upate the profile
	int nbins = m_gpu_bin_data.h_array.Mx * m_gpu_bin_data.h_array.My * m_gpu_bin_data.h_array.Mz;
	uint64_t nthreads = nbins * m_curNmax;
	
	if (m_prof)
		{
		cudaThreadSynchronize();
		// each thread computes 21 flops for each comparison. There are 27*m_avgNmax comparisons per thread.
		// each thread reads 592 bytes: this includes reading its particle position, exclusion,
		// cell lists for neighboring bins and those particle's positions.
		m_prof->pop(m_pdata->getN() * 27 * m_avgNmax * 21, nthreads * 592);
		}
	}
	


//! Test if the list needs updating
bool BinnedNeighborListGPU::needsUpdating(unsigned int timestep)
	{
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
	int result = gpu_nlist_needs_update_check(&pdata, &box, &m_gpu_nlist, r_buffsq);
	//cout << "needs_update: " << result << endl;

	m_pdata->release();

	if (m_prof)
		{
		cudaThreadSynchronize();
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
	}

#ifdef USE_PYTHON
void export_BinnedNeighborListGPU()
	{
	class_<BinnedNeighborListGPU, boost::shared_ptr<BinnedNeighborListGPU>, bases<NeighborList>, boost::noncopyable >
		("BinnedNeighborListGPU", init< boost::shared_ptr<ParticleData>, Scalar, Scalar >())
		;
	}	
#endif
