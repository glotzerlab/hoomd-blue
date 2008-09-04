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

/*! \file gpu_worker_test.cc
	\brief Unit tests for the GPUWorker class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! Name the boost unit test module
#define BOOST_TEST_MODULE GPUWorkerTests
#include "boost_utf_configure.h"

#include <boost/test/floating_point_comparison.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "GPUWorker.h"

using namespace boost;
using namespace std;

#ifdef USE_CUDA

//! boost test case for GPUWorker basic operation
BOOST_AUTO_TEST_CASE( GPUWorker_basic )
	{
	GPUWorker gpu(0);
	
	// try allocating and memcpying some data
	float *d_float;
	float h_float;
	
	// allocate and copy a float to the device
	gpu.call(bind(cudaMalloc, (void **)((void *)&d_float), sizeof(float)));

	h_float = 4.293f;
	gpu.call(bind(cudaMemcpy, d_float, &h_float, sizeof(float), cudaMemcpyHostToDevice));

	// clear the float and copy it back to see if everything worked
	h_float = 0.0f;
	gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost));
	
	BOOST_CHECK_EQUAL(h_float, 4.293f);
	
	gpu.call(bind(cudaFree, d_float));
	}

//! boost test case for GPUWorker error detection
BOOST_AUTO_TEST_CASE( GPUWorker_throw )
	{
	GPUWorker gpu(0);
	
	// try allocating and memcpying some data
	float *d_float;
	float h_float;
	
	// allocate and copy a float to the device
	gpu.call(bind(cudaMalloc, (void **)((void *)&d_float), sizeof(float)));

	h_float = 4.293f;
	// purposefully switch pointers: this should introduce a CUDA error
	// check that an exception is thrown
	BOOST_CHECK_THROW(gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyHostToDevice)), runtime_error);
	
	// the error should be cleared now
	BOOST_CHECK_NO_THROW(gpu.sync());
	
	// test this through sync
	gpu.callAsync(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyHostToDevice));
	BOOST_CHECK_THROW(gpu.sync(), runtime_error);
			
	gpu.call(bind(cudaFree, d_float));
	}

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
