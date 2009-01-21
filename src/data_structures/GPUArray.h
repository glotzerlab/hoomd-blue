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

/*! \file GPUArray.h
	\brief Defines the GPUArray class
*/

#ifndef __GPUARRAY_H__
#define __GPUARRAY_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#else

// GPUArray is likely to be used with various vector types defined in cuda (float4, etc...)
// for builds on systems where CUDA is not available, include copies of the relevant CUDA header
// files for convenience
#include "cudacpu_vector_types.h"
#include "cudacpu_vector_functions.h"

#endif
#include "ExecutionConfiguration.h"

//! Specifies where to acquire the data
struct access_location
	{
	//! The enum
	enum Enum
		{
		host,	//!< Ask to acquire the data on the host
		#ifdef ENABLE_CUDA
		device	//!< Ask to acquire the data on the device
		#endif
		};
	};
	
//! Defines where the data is currently stored
struct data_location
	{
	//! The enum
	enum Enum
		{
		host,		//!< Data was last updated on the host
		#ifdef ENABLE_CUDA
		device,		//!< Data was last updated on the device
		hostdevice	//!< Data is up to date on both the host and device
		#endif
		};
	};
	
//! Sepcify how the data is to be accessed
struct access_mode
	{
	//! The enum
	enum Enum
		{
		read,		//!< Data will be accessed read only
		readwrite,	//!< Data will be accessed for read and write
		overwrite	//!< The data is to be completely overwritten during this aquire
		};
	};
	
template<class T> class GPUArray;

//! Handle to access the data pointer handled by GPUArray
/*! The data in GPUArray is only accessible via ArrayHandle. The pointer is accessible for the lifetime of the 
	ArrayHandle. When the ArrayHandle is destroyed, the GPUArray is notified that the data has been released. This 
	tracking mechanism provides for error checking that will cause code assertions to fail if the data is aquired
	more than once.
	
	ArrayHandle is intended to be used within a scope limiting its use. For example:
	\code
	GPUArray<int> gpu_array(100);
	
		{
		ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
		// use h_handle.data .....
		}
	\endcode
	
	The actual raw pointer \a data should \b NOT be assumed to be the same after the handle is released.
	The pointer may in fact be re-allocated somewhere else after the handle is released and before the next handle
	is acquired.
*/
template<class T> class ArrayHandle
	{
	public:
		//! Aquires the data and sets \a m_data
		inline ArrayHandle(const GPUArray<T>& gpu_array, const access_location::Enum location = access_location::host, 
							const access_mode::Enum mode = access_mode::readwrite);
		//! Notifies the containing GPUArray that the handle has been released
		inline ~ArrayHandle();
		
		T* const data;			//!< Pointer to data
		
	private:
		const GPUArray<T>& m_gpu_array;	//!< Reference to the GPUArray that owns \a data
	};
	
//! Class for managing an array of elements on the GPU mirrored to the CPU
/*!
GPUArray provides a template class for managing the majority of the GPU<->CPU memory usage patterns in 
HOOMD. It represents a single array of elements which is present both on the CPU and GPU. Via 
ArrayHandle, classes can access the array pointers through a handle for a short time. All needed
memory transfers from the host <-> device are handled by the class. For instance, if in the previous call,
the data was acquired for writing on the CPU, this call to acquire for data on the GPU results in the
data being copied to the GPU.
*/
template<class T> class GPUArray
	{
	public:
		//! Constructs a GPUArray
		GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf);
		//! Frees memory
		~GPUArray();
		
		//! Copy constructor
		GPUArray(const GPUArray& from);
		//! = operator
		GPUArray& operator=(const GPUArray& rhs);
		
		//! get the number of elements
		unsigned int getNumElements()
			{
			return m_num_elements;
			}
		
	private:
		unsigned int m_num_elements;			//!< Number of elements
		
		mutable bool m_acquired;				//!< Tracks whether the data has been aquired
		mutable data_location::Enum m_data_location;	//!< Tracks the current location of the data
		ExecutionConfiguration m_exec_conf;		//!< The execution configuration
		
		#ifdef ENABLE_CUDA
		mutable T* d_data;		//!< Pointer to allocated device memory
		#endif
		
		mutable T* h_data;		//!< Pointer to allocated host memory		
		
		//! Acquires the data pointer for use
		inline T* const aquire(const access_location::Enum location, const access_mode::Enum mode, unsigned int gpu) const;
		
		//! Helper function to copy memory from the device to host
		inline void memcpyDeviceToHost() const;
		//! Helper function to copy memory from the host to device
		inline void memcpyHostToDevice() const;
		
		// need to be frineds of all the implementations of ArrayHandle
		friend class ArrayHandle<T>;
	};

//******************************************
// ArrayHandle implementation
// *****************************************

/*! \param gpu_array GPUArray host to the pointer data
	\param location Desired location to access the data
	\param mode Mode to access the data with
*/
template<class T> ArrayHandle<T>::ArrayHandle(const GPUArray<T>& gpu_array, const access_location::Enum location,
												const access_mode::Enum mode) : 
	data(gpu_array.aquire(location, mode, 0)), m_gpu_array(gpu_array)
	{
	}

template<class T> ArrayHandle<T>::~ArrayHandle()
	{
	assert(m_gpu_array.m_acquired);
	m_gpu_array.m_acquired = false;
	}

//******************************************
// GPUArray implementation
// *****************************************

/*! \param num_elements Number of elements to allocate in the array
	\param exec_conf Execution configuration specifying the GPUs on which to allocate memory
*/
template<class T> GPUArray<T>::GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf) : 
	m_num_elements(num_elements), m_acquired(false), m_data_location(data_location::host), m_exec_conf(exec_conf)
	{
	#ifdef ENABLE_CUDA
	// the current implementation only supports a signle GPU
	if (m_exec_conf.gpu.size() > 1)
		{
		std::cerr << std::endl << "GPUArray doesn't support multi-GPU runs yet" << std::endl << std::endl;
		throw std::runtime_error("Error constructing GPUArray");
		}
	#endif
	
	// allocate memory
	#ifdef ENABLE_CUDA
	if (m_exec_conf.gpu.size() > 0)
		{
		m_exec_conf.gpu[0]->call(boost::bind(cudaMallocHost, (void**)((void*)&h_data), m_num_elements*sizeof(T)));
		m_exec_conf.gpu[0]->call(boost::bind(cudaMalloc, (void **)((void *)&d_data), m_num_elements*sizeof(T)));
		}
	#else
	h_data = new T[m_num_elements];
	#endif
	
	// clear memory
	memset(h_data, 0, sizeof(T)*m_num_elements);
	#ifdef ENABLE_CUDA
	if (m_exec_conf.gpu.size() > 0)
		{
		m_exec_conf.gpu[0]->call(boost::bind(cudaMemset, d_data, 0, m_num_elements*sizeof(T)));
		}
	#endif
	}

template<class T> GPUArray<T>::~GPUArray()
	{
	// sanity check
	assert(!m_acquired);
	assert(h_data);
	
	// free memory
	#ifdef ENABLE_CUDA
	assert(d_data);
	if (m_exec_conf.gpu.size() > 0)
		{
		m_exec_conf.gpu[0]->call(boost::bind(cudaFreeHost, h_data));
		m_exec_conf.gpu[0]->call(boost::bind(cudaFree, d_data));
		}
	#else
	delete[] h_data;
	#endif
	
	// set pointers to NULL
	h_data = NULL;
	#ifdef ENABLE_CUDA
	d_data = NULL;
	#endif
	}

template<class T> GPUArray<T>::GPUArray(const GPUArray& from) : m_num_elements(from.m_num_elements), m_acquired(false), 
	m_data_location(data_location::host), m_exec_conf(from.m_exec_conf)
	{	
	// allocate new memory the same size as the data in from
	#ifdef ENABLE_CUDA
	if (m_exec_conf.gpu.size() > 0)
		{
		m_exec_conf.gpu[0]->call(boost::bind(cudaMallocHost, (void**)((void*)&h_data), m_num_elements*sizeof(T)));
		m_exec_conf.gpu[0]->call(boost::bind(cudaMalloc, (void **)((void *)&d_data), m_num_elements*sizeof(T)));
		}
	#else
	h_data = new T[m_num_elements];
	#endif
	
	// copy over the data to the host
		{
		ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
		memcpy(h_data, h_handle.data, sizeof(T)*m_num_elements);
		}
	
	// clear the data on the GPU. Any aquire on the device will result in copying the valid data from the host
	#ifdef ENABLE_CUDA
	if (m_exec_conf.gpu.size() > 0)
		{
		m_exec_conf.gpu[0]->call(boost::bind(cudaMemset, d_data, 0, m_num_elements*sizeof(T)));
		}
	#endif
	}


template<class T> GPUArray<T>& GPUArray<T>::operator=(const GPUArray& rhs)
	{
	if (this != &rhs) // protect against invalid self-assignment
		{
		// sanity check
		assert(!m_acquired);
		assert(h_data);	
		
		// free current memory
		#ifdef ENABLE_CUDA
		if (m_exec_conf.gpu.size() > 0)
			{
			assert(d_data);
			m_exec_conf.gpu[0]->call(boost::bind(cudaFreeHost, h_data));
			m_exec_conf.gpu[0]->call(boost::bind(cudaFree, d_data));
			}
		#else
		delete[] h_data;
		#endif
		
		// copy over basic elements
		m_num_elements = rhs.m_num_elements;
		m_exec_conf = rhs.m_exec_conf;
		
		// initialize state variables
		m_data_location = data_location::host;
		
		// allocate new memory the same size as the data in from
		#ifdef ENABLE_CUDA
		if (m_exec_conf.gpu.size() > 0)
			{
			m_exec_conf.gpu[0]->call(boost::bind(cudaMallocHost, (void**)((void*)&h_data), m_num_elements*sizeof(T)));
			m_exec_conf.gpu[0]->call(boost::bind(cudaMalloc, (void **)((void *)&d_data), m_num_elements*sizeof(T)));
			}
		#else
		h_data = new T[m_num_elements];
		#endif
		
		// copy over the data to the host
			{
			ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
			memcpy(h_data, h_handle.data, sizeof(T)*m_num_elements);
			}
		
		// clear the data on the GPU. Any aquire on the device will result in copying the valid data from the host
		#ifdef ENABLE_CUDA
		if (m_exec_conf.gpu.size() > 0)
			{
			m_exec_conf.gpu[0]->call(boost::bind(cudaMemset, d_data, 0, m_num_elements*sizeof(T)));
			}
		#endif
		}
		
	return *this;
	}

/*! \post All memory on the device is copied to the host array
*/
template<class T> void GPUArray<T>::memcpyDeviceToHost() const
	{
	assert(m_num_elements > 0);
	#ifdef ENABLE_CUDA
	m_exec_conf.gpu[0]->call(boost::bind(cudaMemcpy, h_data, d_data, sizeof(T)*m_num_elements, cudaMemcpyDeviceToHost));
	#endif
	}

/*! \post All memory on the host is copied to the device array
*/	
template<class T> void GPUArray<T>::memcpyHostToDevice() const
	{
	assert(m_num_elements > 0);
	#ifdef ENABLE_CUDA
	m_exec_conf.gpu[0]->call(boost::bind(cudaMemcpy, d_data, h_data, sizeof(T)*m_num_elements, cudaMemcpyHostToDevice));
	#endif
	}	

/*! \param location Desired location to access the data
	\param mode Mode to access the data with
	\param gpu GPU to access the data on (if accessing on the device)

	aquire() is the workhorse of GPUArray. It tracks the internal state variable \a data_location and
	performs all host<->device memory copies as needed during the state changes given the
	specified access mode and location where the data is to be acquired.
	
	aquire() cannot be directly called by the user class. Data must be accessed through ArrayHandle.
*/
template<class T> T* const GPUArray<T>::aquire(const access_location::Enum location, const access_mode::Enum mode, 
	unsigned int gpu) const
	{
	// sanity check
	assert(!m_acquired);
	m_acquired = true;
	
	// first, break down based on where the data is to be acquired
	if (location == access_location::host)
		{
		// then break down based on the current location of the data
		if (m_data_location == data_location::host)
			{
			// the state stays on the host regardles of the access mode
			return h_data;
			}
		#ifdef ENABLE_CUDA
		else if (m_data_location == data_location::hostdevice)
			{
			// finally perform the action baed on the access mode requested
			if (mode == access_mode::read)	// state stays on hostdevice
				m_data_location = data_location::hostdevice;
			else if (mode == access_mode::readwrite)	// state goes to host
				m_data_location = data_location::host;
			else if (mode == access_mode::overwrite)	// state goes to host
				m_data_location = data_location::host;
			else
				{
				std::cerr << std::endl << "Invalid access mode requested" << std::endl << std::endl;
				throw std::runtime_error("Error acquiring data");
				}
				
			return h_data;
			}
		else if (m_data_location == data_location::device)
			{
			// finally perform the action baed on the access mode requested
			if (mode == access_mode::read)
				{
				// need to copy data from the device to the host
				memcpyDeviceToHost();
				// state goes to hostdevice
				m_data_location = data_location::hostdevice;
				}
			else if (mode == access_mode::readwrite)
				{
				// need to copy data from the device to the host
				memcpyDeviceToHost();
				// state goes to host
				m_data_location = data_location::host;
				}
			else if (mode == access_mode::overwrite)
				{
				// no need to copy data, it will be overwritten
				// state goes to host
				m_data_location = data_location::host;	
				}
			else
				{
				std::cerr << std::endl << "Invalid access mode requested" << std::endl << std::endl;
				throw std::runtime_error("Error acquiring data");
				}
				
			return h_data;
			}
		#endif
		else
			{
			std::cerr << std::endl << "Invalid data location state" << std::endl << std::endl;
			throw std::runtime_error("Error acquiring data");
			return NULL;
			}
		}
	#ifdef ENABLE_CUDA
	else if (location == access_location::device)
		{
		// check that a GPU is actually specified
		if (m_exec_conf.gpu.size() == 0)
			{
			std::cerr << std::endl << "Reqesting device aquire, but no GPU in the Execution Configuration" << std::endl << std::endl;
			throw std::runtime_error("Error acquiring data");
			}

		// then break down based on the current location of the data
		if (m_data_location == data_location::host)
			{
			// finally perform the action baed on the access mode requested
			if (mode == access_mode::read)
				{
				// need to copy data to the device
				memcpyHostToDevice();
				// state goes to hostdevice
				m_data_location = data_location::hostdevice;
				}
			else if (mode == access_mode::readwrite)
				{
				// need to copy data to the device
				memcpyHostToDevice();
				// state goes to device
				m_data_location = data_location::device;
				}
			else if (mode == access_mode::overwrite)
				{
				// no need to copy data to the device, it is to be overwritten
				// state goes to device
				m_data_location = data_location::device;
				}
			else
				{
				std::cerr << std::endl << "Invalid access mode requested" << std::endl << std::endl;
				throw std::runtime_error("Error acquiring data");
				}
				
			return d_data;
			}
		else if (m_data_location == data_location::hostdevice)
			{
			// finally perform the action baed on the access mode requested
			if (mode == access_mode::read)	// state stays on hostdevice
				m_data_location = data_location::hostdevice;
			else if (mode == access_mode::readwrite)	// state goes to device
				m_data_location = data_location::device;
			else if (mode == access_mode::overwrite)	// state goes to device
				m_data_location = data_location::device;
			else
				{
				std::cerr << std::endl << "Invalid access mode requested" << std::endl << std::endl;
				throw std::runtime_error("Error acquiring data");
				}
			return d_data;
			}
		else if (m_data_location == data_location::device)
			{
			// the stat stays on the device regardless of the access mode
			return d_data;		
			}
		else
			{
			std::cerr << std::endl << "Invalid data_location state" << std::endl << std::endl;
			throw std::runtime_error("Error acquiring data");
			return NULL;
			}
		}
	#endif
	else
		{
		std::cerr << std::endl << "Invalid location requested" << std::endl << std::endl;
		throw std::runtime_error("Error acquiring data");
		return NULL;
		}
	}

#endif