/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file GPUArray.h
    \brief Defines the GPUArray class
*/

#ifndef __GPUARRAY_H__
#define __GPUARRAY_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#include "gpu_settings.h"
#else

// GPUArray is likely to be used with various vector types defined in cuda (float4, etc...)
// for builds on systems where CUDA is not available, include copies of the relevant CUDA header
// files for convenience
#include "cudacpu_vector_types.h"
#include "cudacpu_vector_functions.h"

#endif
#include "ExecutionConfiguration.h"

#include <iostream>
#include <algorithm>
#include <boost/bind.hpp>

//! Specifies where to acquire the data
struct access_location
    {
    //! The enum
    enum Enum
        {
        host,   //!< Ask to acquire the data on the host
#ifdef ENABLE_CUDA
        device  //!< Ask to acquire the data on the device
#endif
        };
    };

//! Defines where the data is currently stored
struct data_location
    {
    //! The enum
    enum Enum
        {
        host,       //!< Data was last updated on the host
#ifdef ENABLE_CUDA
        device,     //!< Data was last updated on the device
        hostdevice  //!< Data is up to date on both the host and device
#endif
        };
    };

//! Sepcify how the data is to be accessed
struct access_mode
    {
    //! The enum
    enum Enum
        {
        read,       //!< Data will be accessed read only
        readwrite,  //!< Data will be accessed for read and write
        overwrite   //!< The data is to be completely overwritten during this aquire
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
    ... use h_handle.data ...
    }
    \endcode

    The actual raw pointer \a data should \b NOT be assumed to be the same after the handle is released.
    The pointer may in fact be re-allocated somewhere else after the handle is released and before the next handle
    is acquired.

    \ingroup data_structs
*/
template<class T> class ArrayHandle
    {
    public:
        //! Aquires the data and sets \a m_data
        inline ArrayHandle(const GPUArray<T>& gpu_array, const access_location::Enum location = access_location::host,
                           const access_mode::Enum mode = access_mode::readwrite);
        //! Notifies the containing GPUArray that the handle has been released
        inline ~ArrayHandle();
        
        T* const data;          //!< Pointer to data
        
    private:
        const GPUArray<T>& m_gpu_array; //!< Reference to the GPUArray that owns \a data
    };

//! Class for managing an array of elements on the GPU mirrored to the CPU
/*!
GPUArray provides a template class for managing the majority of the GPU<->CPU memory usage patterns in
HOOMD. It represents a single array of elements which is present both on the CPU and GPU. Via
ArrayHandle, classes can access the array pointers through a handle for a short time. All needed
memory transfers from the host <-> device are handled by the class based on the access mode and
location specified when acquiring an ArrayHandle.

GPUArray is fairly advanced, C++ wise. It is a template class, so GPUArray's of floats, float4's,
uint2's, etc.. can be made. It comes with a copy constructor and = operator so you can (expensively)
pass GPUArray's around in arguments or overwite one with another via assignment (inexpensive swaps can be
performed with swap()). The ArrayHandle acquisition method guarantees that every aquired handle will be
released. About the only thing it \b doesn't do is prevent the user from writing to a pointer acquired
with a read only mode.

At a high level, GPUArray encapsulates a single flat data pointer \a T* \a data with \a num_elements
elements, and keeps a copy of this data on both the host and device. When accessing this data through
the construction of an ArrayHandle instance, the \a location (host or device) you wish to access the data
must be specified along with an access \a mode (read, readwrite, overwrite).

When the data is accessed in the same location it was last written to, the pointer is simply returned.
If the data is accessed in a different location, it will be copied before the pointer is returned.

When the data is accessed in the \a read mode, it is assumed that the data will not be written to and
thus there is no need to copy memory the next time the data is aquired somewhere else. Using the readwrite
mode specifies that the data is to be read and written to, necessitating possible copies to the desired location
before the data can be accessed and again before the next access. If the data is to be completely overwritten
\b without reading it first, then an expensive memory copy can be avoided by using the \a overwrite mode.

Data with both 1-D and 2-D representations can be allocated by using the appropriate constructor.
2-D allocated data is still just a flat pointer, but the row width is rounded up to a multiple of
16 elements to facilitate coalescing. The actual allocated width is accessible with getPitch(). Here
is an example of addressing element i,j in a 2-D allocated GPUArray.
\code
GPUArray<int> gpu_array(100, 200, exec_conf);
unsigned int pitch = gpu_array.getPitch();

ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
h_handle.data[i*pitch + j] = 5;
\endcode

A future modification of GPUArray will allow mirroring or splitting the data across multiple GPUs.

\ingroup data_structs
*/
template<class T> class GPUArray
    {
    public:
        //! Constructs a NULL GPUArray
        GPUArray();
        //! Constructs a 1-D GPUArray
        GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf);
        //! Constructs a 2-D GPUArray
        GPUArray(unsigned int width, unsigned int height, const ExecutionConfiguration& exec_conf);
        //! Frees memory
        ~GPUArray();
        
        //! Copy constructor
        GPUArray(const GPUArray& from);
        //! = operator
        GPUArray& operator=(const GPUArray& rhs);
        
        //! Swap the pointers in two GPUArrays
        inline void swap(GPUArray& from);
        
        //! Get the number of elements
        /*!
         - For 1-D allocated GPUArrays, this is the number of elements allocated.
         - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height) allocated
        */
        unsigned int getNumElements() const
            {
            return m_num_elements;
            }
            
        //! Test if the GPUArray is NULL
        bool isNull() const
            {
            return (h_data == NULL);
            }
            
        //! Get the width of the allocated rows in elements
        /*!
         - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the padding added for coalescing)
         - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
        */
        unsigned int getPitch() const
            {
            return m_pitch;
            }
            
        //! Get the number of rows allocated
        /*!
         - For 2-D allocated GPUArrays, this is the height given to the constructor
         - For 1-D allocated GPUArrays, this is the simply 1.
        */
        unsigned int getHeight() const
            {
            return m_height;
            }
    private:
        unsigned int m_num_elements;            //!< Number of elements
        unsigned int m_pitch;                   //!< Pitch of the rows in elements
        unsigned int m_height;                  //!< Number of allocated rows
        
        mutable bool m_acquired;                //!< Tracks whether the data has been aquired
        mutable data_location::Enum m_data_location;    //!< Tracks the current location of the data
        ExecutionConfiguration m_exec_conf;     //!< The execution configuration
        
#ifdef ENABLE_CUDA
        mutable T* d_data;      //!< Pointer to allocated device memory
#endif
        
        mutable T* h_data;      //!< Pointer to allocated host memory
        
        //! Acquires the data pointer for use
        inline T* const aquire(const access_location::Enum location, const access_mode::Enum mode, unsigned int gpu) const;
        
        //! Helper function to allocate memory
        inline void allocate();
        //! Helper function to free memory
        inline void deallocate();
        //! Helper function to clear memory
        inline void memclear();
        
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

template<class T> GPUArray<T>::GPUArray() :
        m_num_elements(0), m_acquired(false), m_data_location(data_location::host), m_exec_conf(false, false, true),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    }

/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Execution configuration specifying the GPUs on which to allocate memory
*/
template<class T> GPUArray<T>::GPUArray(unsigned int num_elements, const ExecutionConfiguration& exec_conf) :
        m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false), m_data_location(data_location::host), m_exec_conf(exec_conf),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear memory
    allocate();
    memclear();
    }

/*! \param width Width of the 2-D array to allocate (in elements)
    \param height Number of rows to allocate in the 2D array
    \param exec_conf Execution configuration specifying the GPUs on which to allocate memory
*/
template<class T> GPUArray<T>::GPUArray(unsigned int width, unsigned int height, const ExecutionConfiguration& exec_conf) :
        m_height(height), m_acquired(false), m_data_location(data_location::host), m_exec_conf(exec_conf),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));
    
    // setup the number of elements
    m_num_elements = m_pitch * m_height;
    
    // allocate and clear memory
    allocate();
    memclear();
    }

template<class T> GPUArray<T>::~GPUArray()
    {
    deallocate();
    }

template<class T> GPUArray<T>::GPUArray(const GPUArray& from) : m_num_elements(from.m_num_elements), m_pitch(from.m_pitch),
        m_height(from.m_height), m_acquired(false), m_data_location(data_location::host), m_exec_conf(from.m_exec_conf),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();
    
    // copy over the data to the new GPUArray
    if (m_num_elements > 0)
        {
        ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
        memcpy(h_data, h_handle.data, sizeof(T)*m_num_elements);
        }
    }

template<class T> GPUArray<T>& GPUArray<T>::operator=(const GPUArray& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // sanity check
        assert(!m_acquired && !rhs.m_acquired);
        
        // free current memory
        deallocate();
        
        // copy over basic elements
        m_num_elements = rhs.m_num_elements;
        m_pitch = rhs.m_pitch;
        m_height = rhs.m_height;
        m_exec_conf = rhs.m_exec_conf;
        
        // initialize state variables
        m_data_location = data_location::host;
        
        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();
        
        // copy over the data to the new GPUArray
        if (m_num_elements > 0)
            {
            ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
            memcpy(h_data, h_handle.data, sizeof(T)*m_num_elements);
            }
        }
        
    return *this;
    }

/*! \param from GPUArray to swap \a this with

    a.swap(b) will result in the equivalent of:
    \code
GPUArray c(a);
a = b;
b = c;
    \endcode

    But it will be done in a super-efficent way by just swapping the internal pointers, thus avoiding all the expensive
    memory deallocations/allocations and copies using the copy constructor and assignment operator.
*/
template<class T> void GPUArray<T>::swap(GPUArray& from)
    {
    // this may work, but really shouldn't be done when aquired
    assert(!m_acquired && !from.m_acquired);
    
    std::swap(m_num_elements, from.m_num_elements);
    std::swap(m_pitch, from.m_pitch);
    std::swap(m_height, from.m_height);
    std::swap(m_acquired, from.m_acquired);
    std::swap(m_data_location, from.m_data_location);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
#endif
    std::swap(h_data, from.h_data);
    }

/*! \pre m_num_elements is set
    \pre pointers are not allocated
    \post All memory pointers needed for GPUArray are allocated
*/
template<class T> void GPUArray<T>::allocate()
    {
    // don't allocate anything if there are zero elements
    if (m_num_elements == 0)
        return;
        
    // sanity check
    assert(h_data == NULL);
    
#ifdef ENABLE_CUDA
    assert(d_data == NULL);
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
        m_exec_conf.gpu[0]->call(boost::bind(cudaHostAllocHack, (void**)((void*)&h_data), m_num_elements*sizeof(T), cudaHostAllocPortable));
        m_exec_conf.gpu[0]->call(boost::bind(cudaMallocHack, (void **)((void *)&d_data), m_num_elements*sizeof(T)));
        }
    else
        {
        h_data = new T[m_num_elements];
        }
#else
    h_data = new T[m_num_elements];
#endif
    }

/*! \pre allocate() has been called
    \post All allocated memory is freed
*/
template<class T> void GPUArray<T>::deallocate()
    {
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;
        
    // sanity check
    assert(!m_acquired);
    assert(h_data);
    
    // free memory
#ifdef ENABLE_CUDA
    if (m_exec_conf.gpu.size() > 0)
        {
        assert(d_data);
        m_exec_conf.gpu[0]->call(boost::bind(cudaFreeHost, h_data));
        m_exec_conf.gpu[0]->call(boost::bind(cudaFree, d_data));
        }
    else
        {
        delete[] h_data;
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

/*! \pre allocate() has been called
    \post All allocated memory is set to 0
*/
template<class T> void GPUArray<T>::memclear()
    {
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;
        
    assert(h_data);
    
    // clear memory
    memset(h_data, 0, sizeof(T)*m_num_elements);
#ifdef ENABLE_CUDA
    if (m_exec_conf.gpu.size() > 0)
        {
        assert(d_data);
        m_exec_conf.gpu[0]->call(boost::bind(cudaMemset, d_data, 0, m_num_elements*sizeof(T)));
        }
#endif
    }


/*! \post All memory on the device is copied to the host array
*/
template<class T> void GPUArray<T>::memcpyDeviceToHost() const
    {
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;
        
#ifdef ENABLE_CUDA
    m_exec_conf.gpu[0]->call(boost::bind(cudaMemcpy, h_data, d_data, sizeof(T)*m_num_elements, cudaMemcpyDeviceToHost));
#endif
    }

/*! \post All memory on the host is copied to the device array
*/
template<class T> void GPUArray<T>::memcpyHostToDevice() const
    {
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;
        
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
            if (mode == access_mode::read)  // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite)    // state goes to host
                m_data_location = data_location::host;
            else if (mode == access_mode::overwrite)    // state goes to host
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
            if (mode == access_mode::read)  // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite)    // state goes to device
                m_data_location = data_location::device;
            else if (mode == access_mode::overwrite)    // state goes to device
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

