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

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include "LJForceComputeThreaded.h"
#include <boost/bind.hpp>
#include <stdexcept>

using namespace std;
using namespace boost;

/*! \file LJForceComputeThreaded.cc
	\brief Contains code for the LJForceComputeThreaded class
*/

/*! \param pdata Particle Data to compute forces on
 	\param nlist Neighborlist to use for computing the forces
	\param r_cut Cuttoff radius beyond which the force is 0
	\param n_threads Number of threads to run simultaneously
	\post In addition to LJForceCompute postconditions, this constructure launches \a n_threads threads. 
		All threads start in a sleeping state. A call to computeForces() will wake them up to start work.
*/
LJForceComputeThreaded::LJForceComputeThreaded(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, unsigned int n_threads) 
	: LJForceCompute(pdata, nlist, r_cut), m_n_threads(n_threads), m_barrier(n_threads)
	{
	// sanity check
	if (m_n_threads == 0)
		throw runtime_error("ZERO threads were requested in LJForceComputeThreaded");
	
	if (m_n_threads > m_pdata->getN())
		throw runtime_error("LJForceComputeThreaded: Requesting more threads than there are particles");

	// initialze condition variables before starting threads
	m_terminate = false;
	m_work_todo_count = 0;
	m_work_done_count = 0;

	// zero the temporary data pointers
	m_tmp_pdata_arrays = NULL;
	m_tmp_nlist = NULL;

	// need to setup the temporary force pointers
	m_tmp_fx.resize(m_n_threads);
	m_tmp_fy.resize(m_n_threads);
	m_tmp_fz.resize(m_n_threads);
	// actual threads will allocate their memory

	// create threads
	// each thread is going to get 1/m_n_threads of the particles in sequence
	unsigned int N = m_pdata->getN();
	unsigned int step = N/m_n_threads;
	// we are working with 1000's of particles here and only tens of threads, 
	// so just give the entire remainder of the division to the first thread
	unsigned int remainder  = N % m_n_threads;
	
	unsigned int start = 0;
	unsigned int end = step + remainder;
	// create the first thread
	// this tricky business of boost::bind creates a function object out of the computeForceThreadFunc of this 
	// class and fires it up as a thread. That is: multiple computeForceThreadFunc's will be running at the same
	// time, all accessing variables in the same instance of this class
	m_threads.create_thread(bind(&LJForceComputeThreaded::computeForceThreadFunc, this, 0, start, end));
	start = end;
	end += step;
	// create the rest of the threads
	for (unsigned int cur_thread = 1; cur_thread < m_n_threads; cur_thread++)
		{
		m_threads.create_thread(bind(&LJForceComputeThreaded::computeForceThreadFunc, this, cur_thread, start, end));
		start = end;
		end += step;
		}	
	}
	
/*! Terminates threads and joins with them
*/
LJForceComputeThreaded::~LJForceComputeThreaded()
	{
	// terminate threads
		{
		mutex::scoped_lock lock(m_work_todo_mutex);
		// set the terminate flag
		m_terminate = true;
		m_work_todo_count = m_n_threads;
		}
	// notify the threads that they are going to die
	m_work_todo_condition.notify_all();
	
	// join the threads in their death
	m_threads.join_all();
	}

/*! \param start First index in the range
	\param end Last index (exclusive) in the range
	\param id Identification number for the thread (0 to m_n_threads)
	
	Handles all thread sleep/work states. Calls computeForceChunk() to perform the actual work of calculating forces.
	This function is only to be called as an independant thread. Upon starting, it immediately starts waiting for work.
	Once it is notified that work is availabie (through m_work_todo_count and conditions), it starts computing forces
	on the specified range \a start to \a end. It is the responsibility of the thread creater to ensure that no
	threads overlap in their ranges.

	Setting m_terminate to true and then setting the m_work_todo_count condition will result in this function returning.
	
	Each thread allocates a temporary work area, sums forces into that area and then performes a reduction into
	\c m_fx, \c m_fy, and \c m_fz so that the third law optimization can be used. It might be possible to get
	faster performace by avoiding the tmp area for full neighborlist modes, but it wouldn't be worth much
	as the third law optimization is faster anyways. Why optimize an unused code path?
*/
void LJForceComputeThreaded::computeForceThreadFunc(unsigned int id, unsigned int start, unsigned int end)
	{
	assert(m_pdata);
	assert(m_tmp_fx.size() == m_n_threads);
	assert(m_tmp_fy.size() == m_n_threads);
	assert(m_tmp_fz.size() == m_n_threads);
	assert(end > start);
	assert(id < m_n_threads);
	assert(m_n_threads >= 1);
	
	// allocate work area
	m_tmp_fx[id] = new Scalar[m_pdata->getN()];
	m_tmp_fy[id] = new Scalar[m_pdata->getN()];
	m_tmp_fz[id] = new Scalar[m_pdata->getN()];	
		
	while (1)
		{
		// wait until there is work to do (signified by a positive m_work_todo_count)
			{
			mutex::scoped_lock lock(m_work_todo_mutex);
			while (m_work_todo_count == 0)
					m_work_todo_condition.wait(lock);
			// decrement the count
			m_work_todo_count--;
			}
		
		// die if we have been flagged to terminate
		if (m_terminate)
			{
			break;
			}
		
		// compute the forces into the 3 work arrays
		computeForceChunk(m_tmp_fx[id], m_tmp_fy[id], m_tmp_fz[id], start, end);
				
		// wait for computations to finish, then complete the force sum
		m_barrier.wait();
		
		// each thread is going to tally up the sum for the particles from start to end
		for (unsigned int i = start; i < end; i++)
			{
			m_fx[i] = m_tmp_fx[0][i];
			m_fy[i] = m_tmp_fy[0][i];
			m_fz[i] = m_tmp_fz[0][i];
			
			for (unsigned int j = 1; j < m_n_threads; j++)
				{
				m_fx[i] += m_tmp_fx[j][i];
				m_fy[i] += m_tmp_fy[j][i];
				m_fz[i] += m_tmp_fz[j][i];
				}
			}

		// increment the done counter so that the boss thread knows when we are done
			{
			mutex::scoped_lock lock(m_work_done_mutex);
			m_work_done_count++;
			}
		m_work_done_condition.notify_all();
		}
		
	delete[] m_tmp_fx[id];
	delete[] m_tmp_fy[id];
	delete[] m_tmp_fz[id];
	}

/*! \post The lennard jones forces are computed for the given timestep. The neighborlist's
 	compute method is called to ensure that it is up to date.

	This method does some houskeeping tasks and then wakes up the worker threads to compute the forces.
	It does not return until all threads have finished their computations.
	
	\param timestep Current time step of the simulation
*/
void LJForceComputeThreaded::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile
	if (m_prof)
		m_prof->push("LJ.Threads");
	
	// depending on the neighborlist settings, we can take advantage of newton's third law
	// to reduce computations at the cost of memory access complexity: set that flag now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	
	// access the neighbor list and store the temporary pointer
	const vector< vector< unsigned int > >& full_list = m_nlist->getList();
	m_tmp_nlist = &full_list;

	// access the particle data and store the temporary pointer
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
	// sanity check
	assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
	m_tmp_pdata_arrays = &arrays;

	if (m_prof)
		m_prof->push("Compute");
	
	// zero the number of calculation counters
		{
		mutex::scoped_lock lock(m_ncalc_mutex);
		m_tmp_ncalc = 0;
		m_tmp_nforcecalc = 0;
		}
	
	// now, it is actually time to wake up the threads
		{
		// setup the work_todo counter so that the threads will start working
		mutex::scoped_lock lock(m_work_todo_mutex);
		m_work_todo_count = m_n_threads;
		}
	m_work_todo_condition.notify_all();
	
	// the threads should now be running, wait on the condition that they are done
		{
		mutex::scoped_lock lock(m_work_done_mutex);
		while (m_work_done_count < m_n_threads)
			m_work_done_condition.wait(lock);
		
		// 0 the done counter for the next call
		m_work_done_count = 0;
		}

	// and that is it. All forces are now calculated
	// FLOPS: 9+12 for each n_calc and an additional 11 for each n_full_calc
		// make that 14 if third_law is 1
	int64_t flops = (9+12)*m_tmp_ncalc + 11* m_tmp_nforcecalc;
	if (third_law)
		flops += 3* m_tmp_nforcecalc;
		
	// memory transferred: 3*sizeof(Scalar) + 2*sizeof(int) for each n_calc
	// plus 3*sizeof(Scalar) for each n_full_calc + another 3*sizeofScalar if third_law is 1
	// PLUS an additional 3*sizeof(Scalar) + sizeof(int) for each particle
	int64_t mem_transfer = 0;
	mem_transfer += (3*sizeof(Scalar) + 2*sizeof(int)) * (m_tmp_ncalc + arrays.nparticles);
	mem_transfer += 3*sizeof(Scalar)*m_tmp_nforcecalc;
	if (third_law)
		mem_transfer += 3*sizeof(Scalar);
	
	m_pdata->release();
	
	#ifdef USE_CUDA
	// the force data is now only up to date on the cpu
	m_data_location = cpu;
	#endif

	if (m_prof)
		{
		m_prof->pop(flops, mem_transfer);
		m_prof->pop();
		}
	}


/*! \param start First index in the range
	\param end Last index (exclusive) the range
	\param fx Work array where forces can be summed independant of other threads: x-component
	\param fy Work array for y-component
	\param fz Work array for z-component
	
	\post Lennard-Jones forces are computed on all particles with indices in the given range.
*/
void LJForceComputeThreaded::computeForceChunk(Scalar * __restrict__ fx, Scalar * __restrict__ fy, Scalar * __restrict__ fz, unsigned int start, unsigned int end)
	{
	// a few sanity checks
	assert(m_tmp_nlist);
	assert(m_tmp_pdata_arrays);
	assert(end > start);
	assert(end <= m_tmp_pdata_arrays->nparticles);
	assert(fx);
	assert(fy);
	assert(fz);
	assert(m_nlist);

	// depending on the neighborlist settings, we can take advantage of newton's third law
	// to reduce computations at the cost of memory access complexity: set that flag now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	
	// get a local copy of the simulation box too
	const BoxDim& box = m_pdata->getBox();
	// sanity check
	assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	

	// create a temporary copy of r_cut sqaured
	Scalar r_cut_sq = m_r_cut * m_r_cut;	 
	
	// precalculate box lenghts for use in the periodic imaging
	Scalar Lx = box.xhi - box.xlo;
	Scalar Ly = box.yhi - box.ylo;
	Scalar Lz = box.zhi - box.zlo;

	// tally up the number of forces calculated in this thread
	int64_t n_calc = 0;
	int64_t n_force_calc = 0;

	// need to start from a zero force
	memset(fx, 0, sizeof(Scalar)*(m_pdata->getN()));
	memset(fy, 0, sizeof(Scalar)*(m_pdata->getN()));
	memset(fz, 0, sizeof(Scalar)*(m_pdata->getN()));	

	// this is copied and pasted from LJForceCompute
	for (unsigned int i = start; i < end; i++)
		{
		// access the particle's position
		Scalar xi = m_tmp_pdata_arrays->x[i];
		Scalar yi = m_tmp_pdata_arrays->y[i];
		Scalar zi = m_tmp_pdata_arrays->z[i];
		
		unsigned int typei = m_tmp_pdata_arrays->type[i];
		// sanity check
		assert(typei < m_pdata->getNTypes());
		
		// access the lj1 and lj2 rows for the current particle type
		Scalar * __restrict__ lj1_row = &(m_lj1[typei*m_ntypes]);
		Scalar * __restrict__ lj2_row = &(m_lj2[typei*m_ntypes]);

		Scalar fxi = 0.0;
		Scalar fyi = 0.0;
		Scalar fzi = 0.0;
		
		// loop over all of the neighbors of this particle
		const vector< unsigned int >& list = (*m_tmp_nlist)[i];
		const unsigned int size = list.size();

		// increment our calculation counter
		n_calc += size;
		for (unsigned int j = 0; j < size; j++)
			{
			unsigned int k = list[j];
			assert(k <= m_tmp_pdata_arrays->nparticles);
				
			// calculate dr
			Scalar dx = xi - m_tmp_pdata_arrays->x[k];
			Scalar dy = yi - m_tmp_pdata_arrays->y[k];
			Scalar dz = zi - m_tmp_pdata_arrays->z[k];
			unsigned int typej = m_tmp_pdata_arrays->type[k];
			// sanity check
			assert(typej < m_pdata->getNTypes());
			
			// apply periodic boundary conditions
			if (dx >= box.xhi)
				dx -= Lx;
			else
			if (dx < box.xlo)
				dx += Lx;

			if (dy >= box.yhi)
				dy -= Ly;
			else
			if (dy < box.ylo)
				dy += Ly;
	
			if (dz >= box.zhi)
				dz -= Lz;
			else
			if (dz < box.zlo)
				dz += Lz;			
			
			// start computing the force
			Scalar rsq = dx*dx + dy*dy + dz*dz;
		
			Scalar fforce = 0.0;
			// only compute the force if the particles are closer than the cuttoff
			if (rsq < r_cut_sq)
				{
				// tally up how many forces we compute
				n_force_calc++;
					
				// compute the force magnitude/r
				Scalar r2inv = Scalar(1.0)/rsq;
				Scalar r6inv = r2inv * r2inv * r2inv;
				Scalar forcelj = r6inv * (lj1_row[typej]*r6inv - lj2_row[typej]);
				fforce = forcelj * r2inv;
				
				// add the force to the particle i
				fxi += dx*fforce;
				fyi += dy*fforce;
				fzi += dz*fforce;
				
				// add the force to particle j if we are using the third law
				if (third_law)
					{
					fx[k] -= dx*fforce;
					fy[k] -= dy*fforce;
					fz[k] -= dz*fforce;
					}
				}
			
			}
		fx[i] += fxi;
		fy[i] += fyi;
		fz[i] += fzi;
		}

	// now that the force computation is done, update the calculation counters
		{
		mutex::scoped_lock lock(m_ncalc_mutex);
		m_tmp_ncalc += n_calc;
		m_tmp_nforcecalc += n_force_calc;
		}
	}
	
#ifdef USE_PYTHON
void export_LJForceComputeThreaded()
	{
	class_<LJForceComputeThreaded, boost::shared_ptr<LJForceComputeThreaded>, bases<LJForceCompute>, boost::noncopyable >
		("LJForceComputeThreaded", init< boost::shared_ptr<ParticleData>, boost::shared_ptr<NeighborList>, Scalar, int >())
		;
	}
#endif

