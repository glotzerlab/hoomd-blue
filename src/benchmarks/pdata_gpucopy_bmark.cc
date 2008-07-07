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

/*! \file pdata_gpucopy_bmark.cc
 	\brief Benchmarks CPU <-> GPU particle data copies
 	\details This is indended a quick test of the performance of the copies
 		to verify that everything is working properly. 
 		
 	\note This executable is only compiled when USE_CUDA is enabled.
 	\ingroup benchmarks
*/

#include "Profiler.h"
#include "ParticleData.h"
#include <iostream>
#include <boost/program_options.hpp>

#include "HOOMDVersion.h"

#include <boost/shared_ptr.hpp>

using namespace boost;
using namespace boost::program_options;
using namespace std;

//! perform cpu -> gpu copy benchmarks
void cpu2gpu_bmark(unsigned int N)
	{
	assert(N > 0);
	int nrepeat = 1000; 
	cout << "---------------------------" << endl;
	cout << "CPU -> GPU copies" << endl;
	
	// construct the particle data
	BoxDim box(Scalar(10.0));
	ParticleData pdata(N, box);
	
	// setup cpu2gpu copies by acquiring read/write on the cpu, then readonly on the gpu
	// to force 1 way copies
	shared_ptr<Profiler> prof(new Profiler());
	pdata.setProfiler(prof);
	ClockSource clk;
	for (int i = 0; i < nrepeat; i++)
		{
		pdata.acquireReadWrite();
		pdata.acquireReadOnlyGPU();
		pdata.release();
		}
	cudaThreadSynchronize();
	
	// measure the final time and output the profile
	int64_t t_total = clk.getTime();
	cout << *prof;
	
	// write out the total time per copy in microseconds
	float time_per_copy = float(t_total/nrepeat)/1e3;
	cout << "Avg time per copy: " << time_per_copy << " us" << endl;
	}
	
//! perform cpu -> gpu copy benchmarks
void gpu2cpu_bmark(unsigned int N)
	{
	assert(N > 0);
	int nrepeat = 1000; 
	cout << "---------------------------" << endl;
	cout << "GPU -> CPU copies" << endl;
	
	// construct the particle data
	BoxDim box(Scalar(10.0));
	ParticleData pdata(N, box);
	
	// setup cpu2gpu copies by acquiring read/write on the cpu, then readonly on the gpu
	// to force 1 way copies
	shared_ptr<Profiler> prof(new Profiler());
	pdata.setProfiler(prof);
	ClockSource clk;
	for (int i = 0; i < nrepeat; i++)
		{
		pdata.acquireReadWriteGPU();
		pdata.acquireReadOnly();
		pdata.release();
		}
	cudaThreadSynchronize();
	
	// measure the final time and output the profile
	int64_t t_total = clk.getTime();
	cout << *prof;
	
	// write out the total time per copy in microseconds
	float time_per_copy = float(t_total/nrepeat)/1e3;
	cout << "Avg time per copy: " << time_per_copy << " us" << endl;
	}
		
//! Parse the command line and run the requested benchmarks 
int main(int argc, char **argv)
	{
	unsigned int N;
	bool cpu2gpu = false;
	bool gpu2cpu = false;	 
	options_description desc("Allowed options");
	desc.add_options()
		("help,h", "Produce help message")
		("nparticles,N", value<unsigned int>(&N)->default_value(64000), "Number of particles")
		("cpu2gpu", value<bool>(&cpu2gpu)->zero_tokens(), "Perform CPU -> GPU benchmarks")
		("gpu2cpu", value<bool>(&gpu2cpu)->zero_tokens(), "Perform GPU -> CPU benchmarks")
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
		exit(1);
		}

	output_version_info(false);

	// if help is specified, print it
	if (vm.count("help"))
		{
		cout << desc;
		exit(1);
		}

	// check for user silly input
	if (N == 0)
		{
		cerr << "Cannot benchmark 0 particles!" << endl;
		exit(1);
		}
	if (!cpu2gpu && !gpu2cpu)
		{
		cerr << "No benchmark specified. Set cpu2gpu or gpu2cpu on the command line." << endl;
		cout << desc;
		exit(1);
		}

	cout << "Running ParticleData CPU <-> GPU copy benchmark on " << N << " particles" << endl;
	if (cpu2gpu)
		cpu2gpu_bmark(N);
	if (gpu2cpu)
		gpu2cpu_bmark(N);

	return 0;
	}
