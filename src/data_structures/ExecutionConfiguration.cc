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

#include "ExecutionConfiguration.h"

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <stdexcept>
#include <iostream>

using namespace std;
using namespace boost;

/*! \file ExecutionConfiguration.cc
	\brief Defines ExecutionConfiguration and related classes
*/

/*! Code previous to the creation of ExecutionConfiguration always used
	CUDA device 0 by default. To maintain continuity, a default constructed
	ExecutionConfiguration will do the same.
*/
ExecutionConfiguration::ExecutionConfiguration()
	{
	#ifdef ENABLE_CUDA
	int dev_count;
	cudaError_t error = cudaGetDeviceCount(&dev_count);
	if (error != cudaSuccess)
		{
		cerr << "***Warning! Error getting CUDA capable device count! Continuing with 0 GPUs." << endl;
		exec_mode = CPU;
		return;
		}
	else
		{
		if (dev_count > 0)
			{
			gpu.push_back(shared_ptr<GPUWorker>(new GPUWorker(0)));
			exec_mode = GPU;
			}
		else
			exec_mode = CPU;
		}
	#else
	exec_mode=CPU;
	#endif
	}

/*! \param mode Execution mode to set (cpu or gpu)
	\param gpu_id GPU to execute on
	
	No GPU is initialized if mode==cpu
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode, unsigned int gpu_id)
	{
	exec_mode = mode;
	
	#ifdef ENABLE_CUDA
	if (exec_mode == GPU)
		{
		int dev_count;
		cudaError_t error = cudaGetDeviceCount(&dev_count);
		if (error != cudaSuccess)
			{
			cerr << endl << "***Error! Error getting CUDA capable device count!" << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			return;
			}
		else
			{
			if ((unsigned int)dev_count > gpu_id)
				gpu.push_back(shared_ptr<GPUWorker>(new GPUWorker(gpu_id)));
			else
				{
				cerr << endl << "***Error! GPU " << gpu_id << " was requested, but only " << dev_count << " was/were found" << endl << endl;
				throw runtime_error("Error initializing execution configuration");
				}
			}
			
		}
	#endif
	}
	
/*! \param mode Execution mode to set (cpu or gpu)
	\param gpu_ids List of GPUs to execute on
	
	No GPU is initialized if mode==cpu
*/
ExecutionConfiguration::ExecutionConfiguration(executionMode mode, const std::vector<unsigned int>& gpu_ids)
	{
	exec_mode = mode;
	
	#ifdef ENABLE_CUDA
	if (exec_mode == GPU)
		{
		int dev_count;
		cudaError_t error = cudaGetDeviceCount(&dev_count);
		if (error != cudaSuccess)
			{
			cerr << endl << "***Error! Error getting CUDA capable device count!" << endl << endl;
			throw runtime_error("Error initializing execution configuration");
			return;
			}
		else
			{
			if (gpu_ids.size() == 0)
				{
				cerr << endl << "***Error! GPU configuration requested with no GPU ids!" << endl << endl;
				throw runtime_error("Error initializing execution configuration");
				return;
				}
				
			for (unsigned int i = 0; i < gpu_ids.size(); i++)
				{
				if ((unsigned int)dev_count > gpu_ids[i])
					gpu.push_back(shared_ptr<GPUWorker>(new GPUWorker(gpu_ids[i])));
				else
					{
					cerr << endl << "***Error! GPU " << gpu_ids[i] << " was requested, but only " << dev_count << " was/were found" << endl << endl;
					throw runtime_error("Error initializing execution configuration");
					}
				}
			}
		}
	#endif
	}
	
#ifdef ENABLE_CUDA
/*! \param file Passed to GPUWorker::setTag
	\param line Passed to GPUWorker::setTag
*/
void ExecutionConfiguration::tagAll(const std::string &file, unsigned int line) const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->setTag(file, line);
	}
	
/*! Calls GPUWorker::sync() for all GPUs in the configuration
*/
void ExecutionConfiguration::syncAll() const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->sync();
	}
	
/*! \param func Passed to GPUWorker::call
*/
void ExecutionConfiguration::callAll(const boost::function< cudaError_t (void) > &func) const
	{
	for (unsigned int i = 0; i < gpu.size(); i++)
		gpu[i]->call(func);
	}
#endif

void export_ExecutionConfiguration()
	{
	scope in_exec_conf = class_<ExecutionConfiguration, boost::shared_ptr<ExecutionConfiguration>, boost::noncopyable >
		("ExecutionConfiguration", init< >())
		.def(init<ExecutionConfiguration::executionMode, unsigned int>())
		.def(init<ExecutionConfiguration::executionMode, vector<unsigned int> >())
		.def_readonly("exec_mode", &ExecutionConfiguration::exec_mode)
		;
		
	enum_<ExecutionConfiguration::executionMode>("executionMode")
		.value("GPU", ExecutionConfiguration::GPU)
		.value("CPU", ExecutionConfiguration::CPU)
	;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
