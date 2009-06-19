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
// Maintainer: joaander

/*! \file GPUWorker.cc
	\brief Code the GPUWorker class
*/

#ifdef ENABLE_CUDA

#include <cuda.h>
#include <boost/bind.hpp>
#include <string>
#include <sstream>
#include <iostream>

#include "GPUWorker.h"

using namespace boost;
using namespace std;

/*! \param dev GPU device number to be passed to cudaSetDevice()
    \param flags Will be passed directly to cudaSetDeviceFlags()
    \param device_arr List of valid devices to be passed to cudaSetValidDevices
    \param len Length of the \a device_arr list to be passed to cudaSetValidDevices
	
	Constructing a GPUWorker creates the worker thread and immeadiately assigns it to 
	a device with cudaSetDevice().
	
	\note: Pass \a dev = -1 in order to skip the cudaSetDevice call. This is intended for use
		with CUDA 2.2 automatic GPU assignment on linux with compute exclusive GPUs.
	\note If \a device_arr is left at the default of NULL, cudaSetValidDevices will not be called
*/
GPUWorker::GPUWorker(int dev, int flags, int *device_arr, int len) 
	: m_exit(false), m_work_to_do(false), m_last_error(cudaSuccess)
	{
	m_tagged_file = __FILE__;
	m_tagged_line = __LINE__;
	m_thread.reset(new thread(bind(&GPUWorker::performWorkLoop, this)));
	
	#if (CUDA_VERSION >= 2020)
	call(bind(cudaSetDeviceFlags, flags));
	call(bind(cudaSetValidDevices, device_arr, len));
	#endif

	if (dev != -1)
		call(bind(cudaSetDevice, dev));
	}
	
/*! Shuts down the worker thread
*/
GPUWorker::~GPUWorker()
	{
	// set the exit condition
		{
		mutex::scoped_lock lock(m_mutex);
		m_work_to_do = true;
		m_exit = true;
		}
	
	// notify the thread there is work to do
	m_cond_work_to_do.notify_one();
	
	// join with the thread
	m_thread->join();
	}
	
		
/*! \param func Function call to execute in the worker thread

	call() executes a CUDA call to in a worker thread. Any function
	with any arguments can be passed in to be queued using boost::bind.
	Examples:
\code
gpu.call(bind(function, arg1, arg2, arg3, ...));
gpu.call(bind(cudaMemcpy, &h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost));
gpu.call(bind(cudaThreadSynchronize));
\endcode
	The only requirement is that the function returns a cudaError_t. Since every
	single CUDA Runtime API function does so, you can call any Runtime API function.
	You can call any custom functions too, as long as you return a cudaError_t representing
	the error of any CUDA functions called within. This is typical in kernel
	driver functions. For example, a .cu file might contain:
\code
__global__ void kernel() { ... }
cudaError_t kernel_driver()
	{
	kernel<<<blocks, threads>>>();
	#ifdef NDEBUG
	return cudaSuccess;
	#else
	cudaThreadSynchronize();
	return cudaGetLastError();
	#endif
	}
\endcode
	It is recommended to just return cudaSuccess in release builds to keep the asynchronous
	call stream going with no cudaThreadSynchronize() overheads.
	
	call() ensures that \a func has been executed before it returns. This is
	desired behavior, most of the time. For calling kernels or other asynchronous
	CUDA functions, use callAsync(), but read the warnings in it's documentation
	carefully and understand what you are doing. Why have callAsync() at all?
	The original purpose for designing GPUWorker is to allow execution on 
	multiple GPUs simultaneously which can only be done with asynchronous calls.
	
	An exception will be thrown if the CUDA call returns anything other than
	cudaSuccess.
*/
void GPUWorker::call(const boost::function< cudaError_t (void) > &func)
	{
	// this mutex lock is to prevent multiple threads from making
	// simultaneous calls. Thus, they can depend on the exception
	// thrown to exactly be the error from their call and not some
	// race condition from another thread
	// making GPUWorker calls to a single GPUWorker from multiple threads 
	// still isn't supported
	mutex::scoped_lock lock(m_call_mutex);

	// call and then sync
	callAsync(func);
	sync();
	//func();
	}

/*! \param func Function to execute inside the worker thread
	
	callAsync is like call(), but  returns immeadiately after entering \a func into the queue. 
	The worker thread will eventually get around to running it. Multiple contiguous
	calls to callAsync() will result in potentially many function calls 
	being queued before any run.

	\warning There are many potential race conditions when using callAsync().
	For instance, consider the following calls:
	\code
gpu.callAsync(bind(cudaMalloc(&d_array, n_bytes)));
gpu.callAsync(bind(cudaMemcpy(d_array, h_array, n_bytes, cudaMemcpyHostToDevice)));
	\endcode
	In this code sequence, the memcpy async call may be created before d_array is assigned
	by the malloc call leading to an invalid d_array in the memcpy. Similar race conditions
	can show up with device to host memcpys. These types of race conditions can be very hard to
	debug, so use callAsync() with caution. Primarily, callAsync() should only be used to call
	cuda functions that are asynchronous normally. If you must use callAsync() on a synchronous
	cuda function (one valid use is doing a memcpy to/from 2 GPUs simultaneously), be
	\b absolutely sure to call sync() before attempting to use the results of the call.
	
	\warning Arguments that are passed into the function call by bind are put into a queue.
	They may have a lifetime longer than that of the caller. If any function performs a
	callAsync and uses pointers to stack variables in the call, sync() \b must be called
	at the end of the function to avoid problems. Similarly, sync() must be called in the
	destructor of any class that passes pointers to member variables into callAsync().
	
	The best practice to avoid problems is to always call sync() at the end of any function 
	that uses callAsync().
*/
void GPUWorker::callAsync(const boost::function< cudaError_t (void) > &func)
	{
	// add the function object to the queue
		{
		mutex::scoped_lock lock(m_mutex);
		m_work_queue.push_back(func);
		m_work_to_do = true;
		}
	
	// notify the threads there is work to do
	m_cond_work_to_do.notify_one();

	// automatically sync() in __DEVICEEMU, --device-emu doesn't seem to be thread safe
	#ifdef _DEVICEEMU
	sync();
	#endif
	}
	
/*! Call sync() to synchronize the master thread with the worker thread.
	After a call to sync() returns, it is guarunteed that all previous
	queued calls (via callAsync()) have been called in the worker thread. 
	
	\note Since many CUDA calls are asynchronous, a call to sync() does not
	necessarily mean that all calls have completed on the GPU. To ensure this,
	one must call() cudaThreadSynchronize():
	\code
gpu.call(bind(cudaThreadSynchronize));
	\endcode

	sync() will throw an exception if any of the queued calls resulted in
	a return value not equal to cudaSuccess.
*/
void GPUWorker::sync()
	{
	// wait on the work done signal
	mutex::scoped_lock lock(m_mutex);
	while (m_work_to_do)
		m_cond_work_done.wait(lock);
		
	// if there was an error
	if (m_last_error != cudaSuccess)
		{
		// build the exception
		cerr << endl << "***Error! " << string(cudaGetErrorString(m_last_error)) << " after " << m_tagged_file << ":" << m_tagged_line << endl << endl;
		runtime_error error("CUDA Error");

		// reset the error value so that it doesn't propagate to continued calls
		m_last_error = cudaSuccess;

		// throw
		throw(error);
		}	
	}


/*! \param file Current file of source code
	\param line Current line of source code

	This is intended to be called worker.setTag(__FILE__, __LINE__). When reporting errors,
	the last file and line tagged will be printed to help identify where the error occured.
*/
void GPUWorker::setTag(const std::string &file, unsigned int line)
	{
	m_tagged_file = file;
	m_tagged_line = line;
	}
	
/*! \internal
	The worker thread spawns a loop that continusously checks the condition variable
	m_cond_work_to_do. As soon as it is signaled that there is work to do with
	m_work_to_do, it processes all queued calls. After all calls are made,
	m_work_to_do is set to false and m_cond_work_done is notified for anyone 
	interested (namely, sync()). During the work, m_exit is also checked. If m_exit
	is true, then the worker thread exits.
*/
void GPUWorker::performWorkLoop()
	{
	bool working = true;
	
	// temporary queue to ping-pong with the m_work_queue
	// this is done so that jobs can be added to m_work_queue while
	// the worker thread is emptying pong_queue
	deque< boost::function< cudaError_t (void) > > pong_queue;
	
	while (working)
		{
		// aquire the lock and wait until there is work to do
			{
			mutex::scoped_lock lock(m_mutex);
			while (!m_work_to_do)
				m_cond_work_to_do.wait(lock);
			
			// check for the exit condition
			if (m_exit)
				working = false;
				
			// ping-pong the queues
			pong_queue.swap(m_work_queue);
			}
			
		// track any error that occurs in this queue
		cudaError_t error = cudaSuccess;
			
		// execute any functions in the queue
		while (!pong_queue.empty())
			{
			// cout << " at " << m_tagged_file << ":" << m_tagged_line << endl;
			cudaError_t tmp_error = pong_queue.front()();
			
			// update error only if it is cudaSuccess
			// this is done so that any error that occurs will propagate through
			// to the next sync()
			if (error == cudaSuccess)
				error = tmp_error;

			pong_queue.pop_front();
			}
		
		// reaquire the lock so we can update m_last_error and 
		// notify that we are done
			{
			mutex::scoped_lock lock(m_mutex);
			
			// update m_last_error only if it is cudaSuccess
			// this is done so that any error that occurs will propagate through
			// to the next sync()
			if (m_last_error == cudaSuccess)
				m_last_error = error;
			
			// notify that we have emptied the queue, but only if the queue is actually empty
			// (call_async() may have added something to the queue while we were executing above)
			if (m_work_queue.empty())
				{
				m_work_to_do = false;
				m_cond_work_done.notify_all();
				}
			}
		}
	}

#endif

