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

/*! \file force_compute_bmark.cc
	\brief Executable for benchmarking all of the various ForceCompute classes
	\details This is intended as a quick test to check the performance of ForceComputes 
		on various machines and after performance tweaks have been made. It allows for a
		number of command line options to change settings, but will most likely just
		be run with the default settings most of the time.
	\ingroup benchmarks
*/

#ifdef WIN32
#include <windows.h>
#endif

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>

#include "BondForceCompute.h"
#include "LJForceCompute.h"
#include "LJForceComputeThreaded.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "SFCPackUpdater.h"

#ifdef USE_CUDA
#include "LJForceComputeGPU.h"
#include "BondForceComputeGPU.h"
#endif

#include "HOOMDVersion.h"

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

using namespace boost::program_options;
using namespace boost;
using namespace std;

// options from command line
//! quite mode: only display time per compute and nothing else
bool quiet = false;
//! number of threads (for applicable computes)
unsigned int nthreads = 1;
//! cutoff radius for pair forces (for applicable computes)
Scalar r_cut = 3.0;
//! buffer radius for pair forces (for applicable computes)
Scalar r_buff = 0.8;
//! block size for calculation (for applicable computes)
unsigned int block_size = 128;
//! number of particles to benchmark
unsigned int N = 100000;
//! Should the particles be sorted with SFCPACK?
bool sort_particles = true;
//! number of seconds to average over
unsigned int nsec = 10;
//! Activate profiling?
bool profile_compute = true;
//! Use half neighborlist?
bool half_nlist = true;
//! Specify packing fraction of particles for the benchmark (if applicable)
Scalar phi_p = 0.2;

//! Helper function to initialize bonds
void init_bond_tables(shared_ptr<BondForceCompute> bf_compute)
	{
	const unsigned int nbonds = 2;
	for (unsigned int j = 0; j < N-nbonds/2; j++)
		{
		for (unsigned int k = 1; k <= nbonds/2; k++)
			bf_compute->addBond(j,j+k);
		}
	}	

//! Initialize the force compute from a string selecting it
shared_ptr<ForceCompute> init_force_compute(const string& fc_name, shared_ptr<ParticleData> pdata, shared_ptr<NeighborList> nlist)
	{
	shared_ptr<ForceCompute> result;
	
	// handle creation of the various lennard=jones computes
	if (fc_name == "LJ")
		result = shared_ptr<ForceCompute>(new LJForceCompute(pdata, nlist, r_cut));
	if (fc_name == "LJ.Threads")
		result = shared_ptr<ForceCompute>(new LJForceComputeThreaded(pdata, nlist, r_cut, nthreads));
	#ifdef USE_CUDA
	if (fc_name == "LJ.GPU")
		{
		shared_ptr<LJForceComputeGPU> tmp = shared_ptr<LJForceComputeGPU>(new LJForceComputeGPU(pdata, nlist, r_cut));
		tmp->setBlockSize(block_size);
		result = tmp;
		}
	#endif
	
	// handle the various bond force computes
	if (fc_name == "Bond")
		{
		shared_ptr<BondForceCompute> tmp = shared_ptr<BondForceCompute>(new BondForceCompute(pdata, 150, 1.0));
		init_bond_tables(tmp);
		result = tmp;
		}
	#ifdef USE_CUDA
	if (fc_name == "Bond.GPU")
		{
		shared_ptr<BondForceComputeGPU> tmp = shared_ptr<BondForceComputeGPU>(new BondForceComputeGPU(pdata, 150, 1.0));
		init_bond_tables(tmp);
		tmp->setBlockSize(block_size);
		result = tmp;
		}
	#endif
		
	return result;
	}
	

//! Initializes the particle data to a random set of particles
shared_ptr<ParticleData> init_pdata()
	{
	RandomInitializer rand_init(N, phi_p, 0.0, "A");
	shared_ptr<ParticleData> pdata(new ParticleData(rand_init));
	if (sort_particles)
		{
		SFCPackUpdater sorter(pdata, Scalar(1.0));
		sorter.update(0);
		}
		
	return pdata;
	}

//! Actually performs the benchmark on the preconstructed force compute
void benchmark(shared_ptr<ForceCompute> fc)
	{
	// initialize profiling if requested
	shared_ptr<Profiler> prof(new Profiler());
	if (profile_compute && !quiet)
		fc->setProfiler(prof);
	
	// timer to count how long we spend at each point
	ClockSource clk;
	int count = 2;

	// do a warmup run so memory allocations don't change the benchmark numbers
	fc->compute(count++);

	int64_t tstart = clk.getTime();
	int64_t tend;
	// do at least one test, then repeat until we get at least 5s of data at this point
	int nrepeat = 0;
	do
		{
		fc->compute(count++);
		nrepeat++;
		tend = clk.getTime();
		} while((tend - tstart) < int64_t(nsec) * int64_t(1000000000) || nrepeat < 5);
	
	// make sure all kernels have been executed when using CUDA
	#ifdef USE_CUDA
	cudaThreadSynchronize();
	#endif
	tend = clk.getTime();
	
	if (!quiet)
		cout << *prof << endl;
	
	double avgTime = double(tend - tstart)/1e9/double(nrepeat);;
	cout << setprecision(7) << avgTime << " s/step" << endl;
	}

//! Parses the command line and runs the benchmark
int main(int argc, char **argv)
	{

	// the name of the ForceCompute to benchmark (gotten from user)
	string fc_name;
	
	options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Produce help message")
		("nparticles,N", value<unsigned int>(&N)->default_value(64000), "Number of particles")
		("phi_p", value<Scalar>(&phi_p)->default_value(0.2), "Volume fraction of particles in test system")
		("r_cut", value<Scalar>(&r_cut)->default_value(3.0), "Cutoff radius for pair force sum")
		("r_buff", value<Scalar>(&r_buff)->default_value(0.8), "Buffer radius for pair force sum")
		("nthreads,t", value<unsigned int>(&nthreads)->default_value(1), "Number of threads to execute (for multithreaded computes)")
		("block_size", value<unsigned int>(&block_size)->default_value(128), "Block size for GPU computes")
		("quiet,q", value<bool>(&quiet)->default_value(false)->zero_tokens(), "Only output time per computation")
		("sort", value<bool>(&sort_particles)->default_value(true), "Sort particles with SFCPACK")
		("profile", value<bool>(&profile_compute)->default_value(true), "Profile GFLOPS and GB/s sustained")
		("half_nlist", value<bool>(&half_nlist)->default_value(true), "Only store 1/2 of the neighbors (optimization for some pair force computes")
		("nsec", value<unsigned int>(&nsec)->default_value(10), "Number of seconds to profile for")
		#ifdef USE_CUDA
		("fc_name,f", value<string>(&fc_name)->default_value("LJ.GPU"), "ForceCompute to benchmark")
		#else
		("fc_name,f", value<string>(&fc_name)->default_value("LJ"), "ForceCompute to benchmark")
		#endif
		;
	
	// parse the command line
	variables_map vm;
	try
		{
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);
		}
	catch (std::logic_error e)
		{
		// print help on error
		cerr << "Error parsing command line: " << e.what() << endl;
		cout << desc;
		cout << "Available ForceComputes are: ";
		cout << "LJ, LJ.Threads, Bond ";
		#ifdef USE_CUDA
		cout << "LJ.GPU, and Bond.GPU" << endl;
		#else
		cout << endl;
		#endif
			
		exit(1);
		}
	
	if (!quiet)
		output_version_info(false);
	
	// if help is specified, print it
	if (vm.count("help"))
		{
		cout << desc;
		exit(1);
		}

	
	
	// initialize the particle data
	if (!quiet)
		cout << "Building particle data..." << endl;
	shared_ptr<ParticleData> pdata = init_pdata();
	
	// initialize the neighbor list
	if (!quiet)
		cout << "Building neighbor list data..." << endl;	
	shared_ptr<NeighborList> nlist(new BinnedNeighborList(pdata, r_cut, r_buff));
	if (half_nlist)
		nlist->setStorageMode(NeighborList::half);
	else
		nlist->setStorageMode(NeighborList::full);
	nlist->setEvery(1000000000);
	nlist->forceUpdate();
	nlist->compute(1);
	
	// count the average number of neighbors
	int64_t neigh_count = 0;
	vector< vector< unsigned int> > list = nlist->getList();
	for (unsigned int i = 0; i < N; i++)
		neigh_count += list[i].size();
	double avgNneigh = double(neigh_count) / double(N);
		
	// initialize the force compute
	shared_ptr<ForceCompute> fc = init_force_compute(fc_name, pdata, nlist);
	if (fc == NULL)
		{
		cerr << "Unrecognized force compute: " << fc_name << endl;
		exit(1);
		}
		
	if (!quiet)
		{
		cout << "Starting benchmarking of: " << fc_name << endl;
		cout << "nsec = " << nsec << " / N = " << N << " / phi_p = "<< phi_p << endl;
		cout << "sort_particles = " << sort_particles << " / half_nlist = " << half_nlist << endl;
		cout << "r_cut = " << r_cut << " / r_buff = " << r_buff << " / avg_n_neigh = " << avgNneigh << endl;
		cout << "nthreads = " << nthreads << " / block_size = " << block_size << endl;
		}
	benchmark(fc);
			
	return 0;
	}
