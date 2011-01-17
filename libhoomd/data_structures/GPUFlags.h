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

/*! \file GPUFlags.h
    \brief Defines the GPUFlags class
*/

#ifndef __GPUFLAGS_H__
#define __GPUFLAGS_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "ExecutionConfiguration.h"

#include <iostream>
#include <stdexcept>

//! Class for managing a small set of flags set on the GPU
/*!
\b Overview:

GPUFlags is an efficient container for a small set of flags. It is optimized for use when a given set of flags is
set via a kernel call on the GPU and then read on the host. The host may also reset the flag to a known value.

\b Implementation details:
Like GPUArray, GPUFlags keeps around an ExecutionConfiguration to keep the GPU alive, and the flags are "mirrored" on the
host and device. In order to optimize for the fastest performance of the host reading flags that were set on the GPU,
the flags are stored in host mapped memory (on devices that support it). GPUFlags inserts synchronization points 1) when
resetFlags() is called to reset the value of the flags (otherwise a running kernel may overwrite the reset value) and
2) when reading the flags on the host with readFlags(). These synchronizations cost about 5-10 microseconds, so do
not read or reset the flags more times than is needed.

\ingroup data_structs
*/
template<class T> class GPUFlags
    {
    public:
        //! Constructs a NULL GPUFlags
        GPUFlags();
        //! Constructs a GPUFlags attached to a GPU
        GPUFlags(boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Frees memory
        ~GPUFlags();
        
        //! Copy constructor
        GPUFlags(const GPUFlags& from);
        //! = operator
        GPUFlags& operator=(const GPUFlags& rhs);
        
        //! Swap the pointers in two GPUFlags
        inline void swap(GPUFlags& from);
        
        //! Test if the GPUFlags is NULL
        bool isNull() const
            {
            return (h_data == NULL);
            }
            
        //! Read the flags on the host
        inline const T readFlags();

        //! Reset the flags on the host
        inline void resetFlags(const T flags);

        //! Get the flags on the device
        T* getDeviceFlags()
            {
            return d_data;
            }
    
    private:
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;    //!< execution configuration for working with CUDA
        bool m_mapped;          //!< Set to true when using host mapped memory
        
#ifdef ENABLE_CUDA
        mutable T* d_data;      //!< Pointer to allocated device memory
#endif
        mutable T* h_data;      //!< Pointer to allocated host memory
        
        //! Helper function to allocate memory
        inline void allocate();
        //! Helper function to free memory
        inline void deallocate();
        //! Helper function to clear memory
        inline void memclear();
        
    };

//******************************************
// GPUFlags implementation
// *****************************************

template<class T> GPUFlags<T>::GPUFlags() :
        m_mapped(false),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    }

/*! \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUFlags<T>::GPUFlags(boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
        m_exec_conf(exec_conf), m_mapped(false),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
#ifdef ENABLE_CUDA 
    // set mapping if requested and supported 
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->dev_prop.canMapHostMemory) 
        m_mapped = true; 
    else 
        m_mapped = false; 
#else 
    m_mapped = false; 
#endif 

    // allocate and clear memory
    allocate();
    memclear();
    }

template<class T> GPUFlags<T>::~GPUFlags()
    {
    deallocate();
    }

template<class T> GPUFlags<T>::GPUFlags(const GPUFlags& from) : m_exec_conf(from.m_exec_conf), m_mapped(false),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();
    
    // copy over the data to the new GPUFlags
    this->resetFlags(from.readFlags);
    }

template<class T> GPUFlags<T>& GPUFlags<T>::operator=(const GPUFlags& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // free current memory
        deallocate();
        
        // copy over basic elements
        m_mapped = rhs.m_mapped;
        m_exec_conf = rhs.m_exec_conf;
        
        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();
        
        // copy over the data to the new GPUFlags
        this->resetFlags(rhs.readFlags());
        }
        
    return *this;
    }

/*! \param from GPUFlags to swap \a this with

    a.swap(b) will result in the equivalent of:
    \code
GPUFlags c(a);
a = b;
b = c;
    \endcode

    But it will be done in a super-efficent way by just swapping the internal pointers, thus avoiding all the expensive
    memory deallocations/allocations and copies using the copy constructor and assignment operator.
*/
template<class T> void GPUFlags<T>::swap(GPUFlags& from)
    {
    std::swap(m_mapped, from.m_mapped);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
#endif
    std::swap(h_data, from.h_data);
    }

/*! \pre pointers are not allocated
    \post All memory pointers needed for GPUFlags are allocated
*/
template<class T> void GPUFlags<T>::allocate()
    {
    // sanity check
    assert(h_data == NULL);

    // allocate memory
#ifdef ENABLE_CUDA
    assert(d_data == NULL);
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        if (m_mapped)
            {
            cudaMallocHost(&h_data, sizeof(T), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&d_data, h_data, 0);
            CHECK_CUDA_ERROR();
            }
        else
            {
            cudaMallocHost(&h_data, sizeof(T));
            cudaMalloc(&d_data, sizeof(T));
            CHECK_CUDA_ERROR();
            }
        }
    else
        {
        h_data = new T;
        }
#else
    h_data = new T;
#endif
    }

/*! \post All allocated memory is freed
*/
template<class T> void GPUFlags<T>::deallocate()
    {
    // don't do anything if the pointers have not been allocated
    if (h_data == NULL)
        return;
        
    // free memory
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
        cudaFreeHost(h_data);
        if (!m_mapped)
            cudaFree(d_data);
        CHECK_CUDA_ERROR();
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
template<class T> void GPUFlags<T>::memclear()
    {
    // don't do anything if nothing is allocated
    if (h_data == NULL)
        return;
        
    assert(h_data);
    
#ifdef ENABLE_CUDA
    // wait for the device to catch up
    if (m_exec_conf && m_exec_conf->isCUDAEnabled() && m_mapped)
        cudaThreadSynchronize();
#endif

    // clear memory
    memset(h_data, 0, sizeof(T));
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled() && !m_mapped)
        {
        assert(d_data);
        cudaMemset(d_data, 0, sizeof(T));
        }
#endif
    }

/*! \returns Current value of the flags
    \note readFlags implicitly syncrhonizes with the GPU execution stream. If there are any previous asynch kernel
    launches that may set the flags, readFlags() will wait until they complete and will return any flags possibly
    set by those kernels.
*/
template<class T> const T GPUFlags<T>::readFlags()
    {
    if (m_mapped)
        {
        // synch to wait for kernels
        cudaThreadSynchronize();
        }
    else
        {
        // memcpy the results to the host
        cudaMemcpy(h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
        }
    
    // return value of flags
    return *h_data;
    }

/*! \param flags Value of flags to set
    \note resetFlags synchronizes with the GPU execution stream. It waits for all prior kernel launches to complete
    before actually resetting the flags so that a possibly executing kernel doesn't unintentionally overwrite the
    intended reset value.
*/
template<class T> void GPUFlags<T>::resetFlags(const T flags)
    {
    if (m_mapped)
        {
        // synch to wait for kernels
        cudaThreadSynchronize();
        // set the flags
        *h_data = flags;
        }
    else
        {
        // set the flags
        *h_data = flags;
        // copy to the device
        cudaMemcpy(d_data, h_data, sizeof(T), cudaMemcpyHostToDevice);
        }
    }

#endif

