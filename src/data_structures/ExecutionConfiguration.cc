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

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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
	
	When the full multi-gpu code is written, ExecutionConfiguration will 
	by default use all of the GPUs found in the device. This will provide the
	user with the default fastest performance with no need for command line options.
*/
ExecutionConfiguration::ExecutionConfiguration()
	{
	#ifdef USE_CUDA
	int dev_count;
	cudaError_t error = cudaGetDeviceCount(&dev_count);
	if (error != cudaSuccess)
		{
		cout << "Error getting CUDA capable device count! Continuing with 0 GPUs." << endl;
		return;
		}
	else
		{
		if (dev_count > 0)
			gpu.push_back(shared_ptr<GPUWorker>(new GPUWorker(0)));
		}
	#endif
	} 
		
