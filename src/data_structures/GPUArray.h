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
// $UR$

/*! \file GPUArray.h
	\brief Defines the GPUArray class
*/

#ifndef __GPUARRAY_H__
#define __GPUARRAY_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#else

// TODO: define vector types

#endif
#include "ExecutionConfiguration.h"

//! Handle to access the data pointer handled by GPUArray
/*! When data is aquired from GPUArray, it is returned in an ArrayHandle.
	The pointer is accessible for the lifetime of the ArrayHandle. When the
	ArrayHandle is destroyed, the GPUArray is notified that the data has been
	released.
*/
template <class T> class ArrayHandle
	{
	private:
		//! Constructs an ArrayHandle
		ArrayHandle(GPUArray<T>& gpu_array, T* const data);
		//! Notifies the containing GPUArray that the handle has been released
		~ArrayHandle();
		
		GPUArray<T>& m_gpu_array;	//!< Reference to source GPUArray
		
		friend GPUArray<T>;
		
	public:
		T* const m_data;			//!< Pointer to data
	};
	
//! Specifies where to acquire the data
enum acquire_location
	{
	host,	//!< Ask to acquire the data on the host
	device	//!< Ask to acquire the data on the device
	};
	
//! Defines where the data is currently stored
enum data_location
	{
	host,		//!< Data was last updated on the host
	device,		//!< Data was last updated on the device
	hostdevice	//!< Data is up to date on both the host and device
	};

//! Class for managing an array of elements on the GPU mirrored to the CPU
/*!
GPUArray provides a template class for managing the majority of the GPU<->CPU memory usage patterns in 
HOOMD. It represents a single array of elements which is present both on the CPU and GPU. Via 
GPUArray::acquire, classes can access the array pointers through a handle for a short time. All needed
memory transfers from the host <-> device are handled by the class. For instance, if in the previous call,
the data was acquired for writing on the CPU, this call to acquire for data on the GPU results in the
data being copied to the GPU.
*/
template <class T> class GPUArray
	{
	public:
		//! Constructs a GPUArray
		GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf);
		//! Frees memory
		~GPUArray();
		
		//! Acquire a handle for read-only data access
		ArrayHandle<T> acquireReadOnly(const acquire_location& location);
		//! Acquire a handle for read-write data access
		ArrayHandle<T>acquireReadWrite(const acquire_location& location);
		//! Acquire a handle to be overwritten
		ArrayHandle<T>acquireOverWrite(const acquire_location& location);
		
	private:
		unsigned int m_num_elements;	//!< Number of elements
		
		bool m_acquired;				//!< Tracks whether the data has been aquired
		data_location m_data_location;	//!< Tracks the current location of the data
		ExecutionConfiguration m_exec_conf;	//!< The execution configuration
		
		#ifdef ENABLE_CUDA
		T* d_data;		//!< Pointer to allocated device memory
		#endif
		
		T* h_data;		//!< Pointer to allocated host memory		
		
		friend ArrayHandle<T*>;
		friend ArrayHandle<const T*>;
	};



//******************************************
// ArrayHandle implementation
// *****************************************

/*! \param gpu_array GPUArray host to the pointer data
	\param data Pointer to the data
*/
template <class T> ArrayHandle::ArrayHandle(GPUArray<T>& gpu_array, T* const data); : 
	m_gpu_array(gpu_array), m_data(data)
	{
	}

template <class T> ArrayHandle::~ArrayHandle()
	{
	assert(m_gpu_array.m_acquired);
	m_gpu_array.m_acquired = false;
	}

//******************************************
// GPUArray implementation
// *****************************************

/*! \param num_elements Number of elements to allocate in the array
*/
template <class T> GPUArray::GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf) : 
	m_num_elements(num_elements), m_acquired(false), m_data_location(host), m_exec_conf(exec_conf)
	{
	// allocate host memory
	#ifdef ENABLE_CUDA
	m_exec_conf.gpu[0]->call(boost::bind(cudaMallocHost, &h_data, num_elements*sizeof(T)));
	m_exec_conf.gpu[0]->call(boost::bind(cudaMalloc, (void **)((void *)&d_data), num_elements*sizeof(T)));
	#else
	h_data = new T[num_elements];
	#endif
	}

template <class T> GPUArray::~GPUArray()
	{
	
	}
		
		//! Acquire a handle for read-only data access
		ArrayHandle<T> acquireReadOnly(const acquire_location& location);
		//! Acquire a handle for read-write data access
		ArrayHandle<T>acquireReadWrite(const acquire_location& location);
		//! Acquire a handle to be overwritten
		ArrayHandle<T>acquireOverWrite(const acquire_location& location);


#endif