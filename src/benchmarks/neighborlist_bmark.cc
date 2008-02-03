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

#ifdef WIN32
#include <windows.h>
#endif

#include <iostream>
#include <iomanip>

#include "NeighborList.h"
#include "BinnedNeighborList.h"

#ifdef USE_CUDA
#include "NeighborListNsqGPU.h"
#include "BinnedNeighborListGPU.h"
#endif

#include "SFCPackUpdater.h"

#include "Initializers.h"
#include <stdlib.h>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

using namespace boost;
using namespace std;

// a simple function for benchmarking the performance of a neighborlist vs the number of particles
// if random is true, the particles are shuffled in memory to present a random memory access pattern 
// to the neighborlist. If it is false, particles near each other in memory are likely to be neighbors
void benchmarkN(int imin, int imax, Scalar rmax, bool sort, boost::function<shared_ptr<NeighborList> (shared_ptr<ParticleData>, Scalar)> create_list)
	{
	// timer to count how long we spend at each point
	ClockSource clk;
	int count = 0;

	
	for (int i = imin; i <= imax; i+=1)
		{
		// construct the particle system	
		RandomInitializer init(i*i*i, Scalar(0.20), 0.0);
		// SimpleCubicInitializer init(40, 1.37);
		shared_ptr<ParticleData> pdata(new ParticleData(init));
		
		shared_ptr<NeighborList> nlist = create_list(pdata, rmax);
	
		if (sort)
			{
			SFCPackUpdater sorter(pdata, 1.0);
			sorter.update(0);
			}

		// do a warmup run so memory allocations don't change the benchmark numbers
		nlist->compute(count++);

		int64_t tstart = clk.getTime();
		int64_t tend;
		// do at least one test, then repeat until we get at least 5s of data at this point
		int nrepeat = 0;
		
		// setup to profile
		shared_ptr<Profiler> prof(new Profiler);
		nlist->setProfiler(prof);
		do
			{
			nlist->compute(count++);
			nrepeat++;
			tend = clk.getTime();
			} while((tend - tstart) < int64_t(5) * int64_t(1000000000) || nrepeat < 5);
		
		//cout << "nrepeat: " << nrepeat << endl;
		double avgTime = double(tend - tstart)/1e9/double(nrepeat);
		cout << i*i*i << " " << setprecision(7) << avgTime << ";" << endl;

		// nlist->printStats();
		
		//cout << *prof;
		}

	}

struct NeighborListCreator
	{
	NeighborListCreator() { }

	shared_ptr<NeighborList> operator()(shared_ptr<ParticleData> pdata, Scalar rmax) const
		{
		shared_ptr<NeighborList> nlist(new NeighborList(pdata, rmax, 0.0));
		nlist->setStorageMode(NeighborList::full);
		return nlist;
		}
	};


/*
#ifdef __SSE__
struct NeighborListSSECreator
	{
	NeighborListSSECreator() { }

	NeighborList *operator()(ParticleData *pdata, Scalar rmax) const
		{
		NeighborList *nlist = new NeighborListSSE(pdata, rmax, 0.0);
		nlist->setStorageMode(NeighborList::full);
		return nlist;
		}
	};
#endif
*/

struct BinnedNeighborListCreator
	{
	BinnedNeighborListCreator() { }
	
	shared_ptr<NeighborList> operator()(shared_ptr<ParticleData> pdata, Scalar rmax) const
		{
		shared_ptr<NeighborList> nlist(new BinnedNeighborList(pdata, rmax, 0.0));
		nlist->setStorageMode(NeighborList::half);
		return nlist;
		}
	};

#ifdef USE_CUDA
struct NsqGPUListCreator
	{
	NsqGPUListCreator() { }
	
	shared_ptr<NeighborList> operator()(shared_ptr<ParticleData> pdata, float rmax) const
		{
		shared_ptr<NeighborList> nlist(new NeighborListNsqGPU(pdata, rmax, 0.0));
		nlist->setStorageMode(NeighborList::full);
		return nlist;
		}
	};

struct BinnedGPUListCreator
	{
	BinnedGPUListCreator() { }
	
	shared_ptr<NeighborList> operator()(shared_ptr<ParticleData> pdata, float rmax) const
		{
		shared_ptr<NeighborList> nlist(new BinnedNeighborListGPU(pdata, rmax, 0.0));
		nlist->setStorageMode(NeighborList::full);
		return nlist;
		}
	};
#endif

int main()
	{
	int imin = 50;
	int imax = 50;
	Scalar r_cut = 4.0;
	
	// quick and dirty for now: construct a ~15000 particle system at phi=0.2
	/*SimpleCubicInitializer init(25, 1.37823);
	ParticleData pdata(init);

	// shuffle arrays so that there are no patterns in the data
	shuffle_particles(&pdata);

	Profiler prof;
	BinnedNeighborList nlist(&pdata, 3.7);
	nlist.setProfiler(&prof);
	nlist.setSSE(false);
	nlist.setStorageMode(NeighborList::full);

	for (unsigned int i = 0; i < pdata.getN() - 1; i++)
		nlist.addExclusion(i,i+1);

	for (int i = 0; i < 100; i++)
		nlist.compute(i);
	cout << prof;*/

	/*Profiler prof2;
	nlist.setProfiler(&prof2);
	nlist.setStorageMode(NeighborList::half);
	for (int i = 0; i < 100; i++)
		nlist.compute(i+1000);

	cout << prof2;*/

	/*cout << "data_nsq_cpu_unsorted = [" << endl;
	benchmarkN(imin, 20, r_cut, false, NeighborListCreator());
	cout << "];" << endl;

	cout << "data_nsq_cpu_sorted = [" << endl;
	benchmarkN(imin, 20, r_cut, true, NeighborListCreator());
	cout << "];" << endl;
	
	#ifdef __SSE__
	cout << "data_nsq_sse_unsorted = [" << endl;
	benchmarkN(imin, 20, r_cut, false, NeighborListSSECreator());
	cout << "];" << endl;
	
	cout << "data_nsq_sse_sorted = [" << endl;
	benchmarkN(imin, 20, r_cut, true, NeighborListSSECreator());
	cout << "];" << endl;
	#endif*/
	
	/*cout << "data_n_cpu_unsorted = [" << endl;
	benchmarkN(imin, imax, r_cut, false, BinnedNeighborListCreator());
	cout << "];" << endl;*/

	/*cout << "data_n_cpu_sorted = [" << endl;
	benchmarkN(imin, imax, r_cut, true, BinnedNeighborListCreator());
	cout << "];" << endl;*/
	

	#ifdef USE_CUDA
	
	/*cout << "data_nsq_gpu_unsorted = [" << endl;
	benchmarkN(imin, 30, r_cut, false, NsqGPUListCreator());
	cout << "];" << endl;
	
	cout << "data_nsq_gpu_sorted = [" << endl;
	benchmarkN(imin, 30, r_cut, true, NsqGPUListCreator());
	cout << "];" << endl;*/
	
	//cout << "data_n_gpu_unsorted = [" << endl;
	//benchmarkN(imin, imax, r_cut, false, BinnedGPUListCreator());
	//cout << "];" << endl;
	
	//cout << "data_n_gpu_sorted = [" << endl;
	benchmarkN(imin, imax, r_cut, true, BinnedGPUListCreator());
	//cout << "];" << endl;
	cout << endl;
	#endif
	
	return 0;
	}

