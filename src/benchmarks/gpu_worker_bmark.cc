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

/*! \file gpu_worker_bmark.cc
	\brief Executable for benchmarking the GPUWorker class
	\details This is just a quick and dirty file to test the latency and overhead of
		GPUWorker.
	\ingroup benchmarks
*/


#include <cuda_runtime.h>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>

#include "ClockSource.h"
#include "GPUWorker.h"

using namespace boost;
using namespace std;

//! Tests a low latency operation (memcpy of 4 bytes) through the worker
void latency_test_worker()
	{
	cout << "GPUWorker latency test" << endl;
	GPUWorker gpu(0);
	
	// try allocating and memcpying some data
	float *d_float;
	float h_float;
	
	// allocate and copy a float to the device
	gpu.call(bind(cudaMalloc, (void **)((void *)&d_float), sizeof(float)));
	h_float = 4.293f;
	gpu.call(bind(cudaMemcpy, d_float, &h_float, sizeof(float), cudaMemcpyHostToDevice));
	
	// clear the float and copy it back
	h_float = 0.0f;
	gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost));
	
	// now that everything is warmed up, make lots of these little calls and time them
	gpu.call(bind(cudaThreadSynchronize));
	
	ClockSource clk;
	const int n_iters = 1000;
	for (int i = 0; i < n_iters; i++)
		gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost));
	
	gpu.call(bind(cudaThreadSynchronize));
	
	uint64_t time = clk.getTime();
	
	cout << "Time per call " << double(time) / 1e3 / double(n_iters) << " us" << endl;
	
	gpu.call(bind(cudaFree, d_float));
	}

//! Tests a low latency operation (mempcy of 4 bytes) without GPUWorker
void latency_test_standard()
	{
	cout << "Standard latency test" << endl;
	cudaSetDevice(0);
	
	// try allocating and memcpying some data
	float *d_float;
	float h_float;
	
	// allocate and copy a float to the device
	cudaMalloc((void **)((void *)&d_float), sizeof(float));
	h_float = 4.293f;
	cudaMemcpy(d_float, &h_float, sizeof(float), cudaMemcpyHostToDevice);
	
	// clear the float and copy it back
	h_float = 0.0f;
	cudaMemcpy(&h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost);
	
	// now that everything is warmed up, make lots of these little calls and time them
	cudaThreadSynchronize();
	
	ClockSource clk;
	const int n_iters = 1000;
	for (int i = 0; i < n_iters; i++)
		cudaMemcpy(&h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaThreadSynchronize();
	uint64_t time = clk.getTime();
	
	cout << "Time per call " << double(time) / 1e3 / double(n_iters) << " us" << endl;
	
	cudaFree(d_float);
	}

// realistic test

//! kerenl from gpu_worker_bmark_kernel.cu
extern "C" cudaError_t kernel_caller(int *data, int N);

const int N = 512*1000;

//! Tests a realistic series of kernel calls without GPUWorker
void realistic_test_standard()
	{
	cout << "Standard realistic test" << endl;
	cudaSetDevice(0);

	// allocate data
	int *d_data;
	cudaMalloc((void**)((void*)&d_data), sizeof(int)*N);
	// warm up kernel
	kernel_caller(d_data, N);

	// start timing
	cudaThreadSynchronize();
	ClockSource clk;
	const int n_iters = 100;
	int val;
	for (int i = 0; i < n_iters; i++)
		{
		kernel_caller(d_data, N);
		if (i % 10 == 0)
			cudaMemcpy(&val, d_data, sizeof(int), cudaMemcpyDeviceToHost);
		}
	
	cudaThreadSynchronize();
	uint64_t time = clk.getTime();
	
	cout << "Time per step " << double(time) / 1e3 / double(n_iters) << " us" << endl;
	}	

//! Tests a realistic series of kernel calls with GPUWorker
void realistic_test_worker()
	{
	cout << "GPUWorker realistic test" << endl;
	GPUWorker gpu(0);

	// allocate data
	int *d_data;
	gpu.call(bind(cudaMalloc, (void**)((void*)&d_data), sizeof(int)*N));
	// warm up kernel
	gpu.callAsync(bind(kernel_caller, d_data, N));

	// start timing
	gpu.call(bind(cudaThreadSynchronize));
	ClockSource clk;
	const int n_iters = 100;
	int val;
	for (int i = 0; i < n_iters; i++)
		{
		gpu.callAsync(bind(kernel_caller, d_data, N));
		if (i % 10 == 0)
			gpu.call(bind(cudaMemcpy, &val, d_data, sizeof(int), cudaMemcpyDeviceToHost));
		}
	
	gpu.call(bind(cudaThreadSynchronize));
	uint64_t time = clk.getTime();
	
	cout << "Time per step " << double(time) / 1e3 / double(n_iters) << " us" << endl;
	}	

//! Mutex so couts don't overlap
mutex iomutex;

//! realistic test peer thread for multi-gpu perf test
void realistic_mgpu_peer(int dev)
	{
		{
		mutex::scoped_lock lock(iomutex);
		cout << "Peer-based mgpu test (GPU " << dev << ")" << endl;
		}
	cudaSetDevice(dev);

	// allocate data
	int *d_data;
	cudaMalloc((void**)((void*)&d_data), sizeof(int)*N);
	// warm up kernel
	kernel_caller(d_data, N);

	// start timing
	cudaThreadSynchronize();
	ClockSource clk;
	const int n_iters = 500;
	int val;
	for (int i = 0; i < n_iters; i++)
		{
		kernel_caller(d_data, N);
		if (i % 10 == 0)
			cudaMemcpy(&val, d_data, sizeof(int), cudaMemcpyDeviceToHost);
		}
	
	cudaThreadSynchronize();
	uint64_t time = clk.getTime();
	
		{
		mutex::scoped_lock lock(iomutex);
		cout << "Time per step (GPU " << dev << ") " << double(time) / 1e3 / double(n_iters) << " us" << endl;
		}
	}	

//! Creates 2 peer threads for perf testing 2 gpus: don't call unless there are at least 2 GPUs in the system
void realistic_mgpu_peer()
	{
	thread thr1(bind(realistic_mgpu_peer, 0));
	thread thr2(bind(realistic_mgpu_peer, 1));
	thr1.join();
	thr2.join();
	}

//! Creates 2 GPUWorkers and runs the multi-gpu perf test: don't call unless there are ast least 2 GPUs in the system
void realistic_mgpu_master_slave()
	{
	cout << "Master/slave-based mgpu test" << endl;
	GPUWorker gpu0(0);
	GPUWorker gpu1(1);

	// allocate data
	int *d_data0;
	gpu0.call(bind(cudaMalloc, (void**)((void*)&d_data0), sizeof(int)*N));
	int *d_data1;
	gpu1.call(bind(cudaMalloc, (void**)((void*)&d_data1), sizeof(int)*N));
	
	// warm up kernel
	gpu0.callAsync(bind(kernel_caller, d_data0, N));
	gpu1.callAsync(bind(kernel_caller, d_data1, N));


	// start timing
	gpu0.call(bind(cudaThreadSynchronize));
	gpu1.call(bind(cudaThreadSynchronize));

	ClockSource clk;
	const int n_iters = 500;
	int val0, val1;
	for (int i = 0; i < n_iters; i++)
		{
		gpu0.callAsync(bind(kernel_caller, d_data0, N));
		gpu1.callAsync(bind(kernel_caller, d_data1, N));
		if (i % 10 == 0)
			{
			gpu0.callAsync(bind(cudaMemcpy, &val0, d_data0, sizeof(int), cudaMemcpyDeviceToHost));
			gpu1.callAsync(bind(cudaMemcpy, &val1, d_data1, sizeof(int), cudaMemcpyDeviceToHost));
			// ensure that the memory read was made
			gpu0.sync();
			gpu1.sync();
			}
		}
	
	gpu0.call(bind(cudaThreadSynchronize));
	gpu1.call(bind(cudaThreadSynchronize));
	uint64_t time = clk.getTime();
	
	cout << "Time per step " << double(time) / 1e3 / double(n_iters) << " us" << endl;
	}	

int main(int argc, char **argv)
	{
	latency_test_worker();
	cout << endl;
	latency_test_standard();
	cout << endl;
	realistic_test_standard();
	cout << endl;
	realistic_test_worker();

	int dev_count;
	cudaGetDeviceCount(&dev_count);
	if (dev_count >= 2)
		{
		cout << endl;
		realistic_mgpu_peer();
		cout << endl;
		realistic_mgpu_master_slave();
		}
	}
