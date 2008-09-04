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

#ifndef __EXECUTION_CONFIGURATION__
#define __EXECUTION_CONFIGURATION__

#include "GPUWorker.h"

#include <vector>
#include <boost/shared_ptr.hpp>

/*! \file ExecutionConfiguration.h
	\brief Declares ExecutionConfiguration and related classes
*/

//! Defines the execution configuration for the simulation
/*! \ingroup data_structs
	This is just a skeleton of what ExecutionConfiguration will eventually be.
	It currently only contains a list of GPUWorkers that the system is
	to use in executing the simulation. Current code only uses one GPU,
	so the vector is used for future expansion purposes.
	
	The execution mode is specified in exec_mode. This is only to be taken as a hint,
	different compute classes are free to execute on either the CPU or GPU. The only
	requirement is that those executing on the GPU must use the gpu workers in the vector.
*/
struct ExecutionConfiguration
	{
	//! Simple enum for the execution modes
	enum executionMode
		{
		GPU,	//!< Execute on the GPU
		CPU	//!< Execute on the CPU
		};
	
	//! Default constructor
	ExecutionConfiguration();
	
	//! Single GPU selection constructor
	ExecutionConfiguration(executionMode mode, unsigned int gpu_id);
	
	//! Multi GPU selection constructor
	ExecutionConfiguration(executionMode mode, const std::vector<unsigned int>& gpu_ids);
	
	executionMode exec_mode;	//!< Execution mode specified in the constructor
	#ifdef USE_CUDA
	std::vector< boost::shared_ptr<GPUWorker> > gpu;	//!< GPUs to execute on
	#endif
	};

#ifdef USE_PYTHON
//! Exports ExecutionConfiguration to python
void export_ExecutionConfiguration();
#endif
	
#endif
