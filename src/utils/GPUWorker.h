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

/*! \file GPUWorker.h
	\brief Defines the GPUWorker class
*/

// only compile if ENABLE_CUDA is enabled
#ifdef ENABLE_CUDA

#ifndef __GPUWORKER_H__
#define __GPUWORKER_H__

#include <deque>
#include <stdexcept>
#include <vector>

#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/scoped_ptr.hpp>

#include <cuda_runtime_api.h>

//! Implements a worker thread controlling a single GPU
/*! CUDA requires one thread per GPU in multiple GPU code. It is not always
	convenient to write multiple-threaded code where all threads are peers.
	Sometimes, a master/slave approach can be the simplest and quickest to write.
	
	GPUWorker provides the underlying worker threads that a master/slave
	approach needs to execute on multiple GPUs. It is designed so that 
	a \b single thread can own multiple GPUWorkers, each of whom execute on 
	their own GPU. The master thread can call any CUDA function on that GPU
	by passing a bound boost::function into call() or callAsync(). Internally, these
	calls are executed inside the worker thread so that they all share the same
	CUDA context.
	
	On construction, a GPUWorker is automatically associated with a device. You
	pass in an integer device number which is used to call cudaSetDevice() 
	in the worker thread. 
	
	After the GPUWorker is constructed, you can make calls on the GPU
	by submitting them with call(). To queue calls, use callAsync(), but
	please read carefully and understand the race condition warnings before 
	using callAsync(). sync() can be used to synchronize the master thread
	with the worker thread. If any called GPU function returns an error,
	call() (or the sync() after a callAsync()) will throw a std::runtime_error.
	
	To share a single GPUWorker with multiple objects, use boost::shared_ptr.
\code
boost::shared_ptr<GPUWorker> gpu(new GPUWorker(dev));
gpu->call(whatever...)
SomeClass cls(gpu);
// now cls can use gpu to execute in the same worker thread as everybody else
\endcode	
	
	\warning A single GPUWorker is intended to be used by a \b single master thread
	(though master threads can use multiple GPUWorkers). If a single GPUWorker is
	shared amoung multiple threads then ther \e should not be any horrible consequences.
	All tasks will still be exected in the order in which they
	are recieved, but sync() becomes ill-defined (how can one synchronize with a worker that
	may be receiving commands from another master thread?) and consequently all synchronous
	calls via call() \b may not actually be synchronous leading to weird race conditions for the
	caller. Then againm calls via call() \b might work due to the inclusion of a mutex lock:
	still, multiple threads calling a single GPUWorker is an untested configuration.
	Use at your own risk.

	\note GPUWorker works in both Linux and Windows (tested with VS2005). However,
	in Windows, you need to define BOOST_BIND_ENABLE_STDCALL in your project options
	in order to be able to call CUDA runtime API functions with boost::bind.
	
	\ingroup utils
*/
class GPUWorker
	{
	public:
		//! Creates a worker thread and ties it to a particular gpu \a dev
		GPUWorker(int dev, int flags=0, int *device_arr=NULL, int len=0);
		
		//! Destructor
		~GPUWorker();
		
		//! Makes a synchronous function call executed by the worker thread
		void call(const boost::function< cudaError_t (void) > &func);
		
		//! Queues an asynchronous function call to be executed by the worker thread
		void callAsync(const boost::function< cudaError_t (void) > &func);

		//! Blocks the calling thread until all queued calls have been executed
		void sync();

		//! Tag the current location in the code
		void setTag(const std::string &file, unsigned int line);
	
	private:
		//! Flag to indicate the worker thread is to exit
		bool m_exit;
		
		//! Flag to indicate there is work to do
		bool m_work_to_do;
		
		//! Error from last cuda call
		cudaError_t m_last_error;

		//! Tagged file
		std::string m_tagged_file;

		//! Tagged line
		unsigned int m_tagged_line;
		
		//! The queue of function calls to make
		std::deque< boost::function< cudaError_t (void) > > m_work_queue;
		
		//! Mutex for accessing m_exit, m_work_queue, m_work_to_do, and m_last_error
		boost::mutex m_mutex;
		
		//! Mutex for syncing after every operation
		boost::mutex m_call_mutex;
		
		//! Condition variable to signal m_work_to_do = true
		boost::condition m_cond_work_to_do;
		
		//! Condition variable to signal m_work_to_do = false (work is complete)
		boost::condition m_cond_work_done;
		
		//! Thread
		boost::scoped_ptr<boost::thread> m_thread;
		
		//! Worker thread loop
		void performWorkLoop();
	};
		
		
#endif
#endif
