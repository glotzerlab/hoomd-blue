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

#include "LJForceCompute.h"
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

/*! \file LJForceComputeThreaded.h
	\brief Declares a class for computing lennard jones forces using multiple threads
*/

#ifndef __LJFORCECOMPUTETHREADED_H__
#define __LJFORCECOMPUTETHREADED_H__

//! Computes lennard jones forces on each particle using multiple threads
/*! Computed forces should be identical to those in LJForceCompute (up to floating point rounding errors),
	but this class will be much faster on a multiple core/multiple processor system. The number of worker 
	threads this class activates at once is specified to the constructor in the argument \a n_threads. 
	To obtain the best performance possible, it is not recommended to set \a n_threads higher than the 
	number of processor cores in the system.
	
	LJForceComputeThreaded can be used in exactly the same manner as LJForceCompute.

	Implementation details (for developer information):
	Each thread will compute the forces on a subset of the particles. All threads will be run using
	boost::threads to run the executeThreads() member of this class. Initially, threads will start up 
	in a waiting state. When computeForces() is called, m_work_to_do is set to the number of threads started
	and a condition variable is used to notify the threads to start their work. Each thread that wakes up on the 
	condition will decrement m_work_todo_count. If m_terminate is true, then the work the threads do is exit. Otherwise
	they fire up the force computation and handle their particles. If Newton's third law is being used, forces
	are summed in temporary arrays, a barrier is set, and then the multiple force computations are summed into 
	m_fx, m_fy, and m_fz. When done, a thread increments m_work_done in a condition variable: when computeForces()
	wakes up on the condition work_done_count == n_threads, the forces have been computed and it can return.

	Due to the design of ParticleData, it can only be acquired once, so computeForces acquires the particle data 
	and puts the pointer to the arrays into a temporary location for the threads to read: m_tmp_pdata_arrays. Similarly
	the neighborlist is stored in m_tmp_nlist. These tmp structures are ONLY valid from the time computeForces()
	launches the threads to the time that they finish their work.
	\ingroup computes
*/
class LJForceComputeThreaded : public LJForceCompute
	{
	public:
		//! Constructs the compute
		LJForceComputeThreaded(boost::shared_ptr<ParticleData> pdata, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, unsigned int n_threads);
		
		//! Destructor
		virtual ~LJForceComputeThreaded();
		
	protected:
		bool m_terminate;				//!< Flag to tell the threads to terminate
		unsigned int m_n_threads;		//!< Count of the number of threads running

		unsigned int m_work_todo_count;	//!< Count of the threads that still have work to do
		boost::condition m_work_todo_condition;	//!< Condition variable for m_work_todo_count
		boost::mutex m_work_todo_mutex;	//!< Mutex for m_work_todo_count

		unsigned int m_work_done_count;	//!< Count of the threads that have finished their work
		boost::condition m_work_done_condition;	//!< Condition variable for m_work_done_count
		boost::mutex m_work_done_mutex;	//!< Mutex for m_work_done_count;

		boost::thread_group m_threads;	//!< Actual running threads

		boost::barrier m_barrier;	//!< Barrier for synchronizing threads
				
		ParticleDataArraysConst const * m_tmp_pdata_arrays;	//!< Temporary pointer for the particle data
		std::vector< std::vector< unsigned int > >const * m_tmp_nlist;	//!< Temporary pointer for the neighbor list

		std::vector<Scalar *> m_tmp_fx;	//!< Per thread fx for summing forces before reduction
		std::vector<Scalar *> m_tmp_fy;	//!< Per thread fy for summing forces before reduction
		std::vector<Scalar *> m_tmp_fz;	//!< Per thread fz for summing forces before reduction
				
		boost::mutex m_ncalc_mutex;	//!< Mutex for controlling access to m_tmp_calc and m_tmp_nforcecalc
		int64_t m_tmp_ncalc;	//!< Location where threads can sum their number of calculations
		int64_t m_tmp_nforcecalc;	//!< Location where threads can sum their number of force calculations
				
		//boost::mutex m_iomutex;	//!< Mutex for handling couts when they are needed for debugging

		//! Thread function for handling all synchronization and force computation
		void computeForceThreadFunc(unsigned int id, unsigned int start, unsigned int end);

		//! Low level function for computing some of the forces
		void computeForceChunk(Scalar * __restrict__ fx, Scalar * __restrict__ fy, Scalar * __restrict__ fz, unsigned int start, unsigned int end);

		//! Fires up the threads to compute the forces
		virtual void computeForces(unsigned int timestep);
	};
	
#ifdef USE_PYTHON
//! Exports the LJForceComputeThreaded class to python
void export_LJForceComputeThreaded();
#endif

#endif

