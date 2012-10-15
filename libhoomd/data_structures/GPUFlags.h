/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file GPUFlags.h
    \brief Defines the GPUFlags class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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

        //! Read the flags on the host (threaded version)
        /*! \param thread_id The thread from which this method is called
         */
        inline const T readFlagsThread(unsigned int thread_id);

        //! Reset the flags on the host
        inline void resetFlags(const T flags);

        //! Reset the flags on the host (threaded version)
        /*! \param thread_id The thread from which this method is called
         */
        inline void resetFlagsThread(const T flags, unsigned int thread_id);

#ifdef ENABLE_CUDA
        //! Get the flags on the device
        T* getDeviceFlags()
            {
            return d_data;
            }
#endif
    
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
        m_exec_conf->useContext();
        if (m_mapped)
            {
            cudaHostAlloc(&h_data, sizeof(T), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&d_data, h_data, 0);
            CHECK_CUDA_ERROR();
            }
        else
            {
            cudaHostAlloc(&h_data, sizeof(T), cudaHostAllocDefault);
            cudaMalloc(&d_data, sizeof(T));
            CHECK_CUDA_ERROR();
            }
        m_exec_conf->releaseContext();
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
        m_exec_conf->useContext();
        cudaFreeHost(h_data);
        if (!m_mapped)
            cudaFree(d_data);
        CHECK_CUDA_ERROR();
        m_exec_conf->releaseContext();
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
        {
        m_exec_conf->useContext();
        cudaThreadSynchronize();
        m_exec_conf->releaseContext();
        }
#endif

    // clear memory
    memset(h_data, 0, sizeof(T));
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled() && !m_mapped)
        {
        assert(d_data);
        m_exec_conf->useContext();
        cudaMemset(d_data, 0, sizeof(T));
        m_exec_conf->releaseContext();
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
#ifdef ENABLE_CUDA
    if (m_mapped)
        {
        // synch to wait for kernels
        m_exec_conf->useContext();
        cudaThreadSynchronize();
        m_exec_conf->releaseContext();
        }
    else
        {
        // memcpy the results to the host
        m_exec_conf->useContext();
        cudaMemcpy(h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
        m_exec_conf->releaseContext();
        }
#endif    

    // return value of flags
    return *h_data;
    }

template<class T> inline const T GPUFlags<T>::readFlagsThread(unsigned int thread_id)
    {
#ifdef ENABLE_CUDA
    cudaEvent_t ev = m_exec_conf->getThreadEvent(thread_id);
    cudaStream_t stream = m_exec_conf->getThreadStream(thread_id);

    m_exec_conf->useContext();
    if (m_mapped)
        {
        // synch to wait for kernels
        cudaEventRecord(ev,stream);
        }
    else
        {
        // async memcpy the results to the host
        cudaMemcpyAsync(h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost,stream);
        cudaEventRecord(ev,stream);
        }
    cudaEventSynchronize(ev);
    m_exec_conf->releaseContext();
#endif    

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
#ifdef ENABLE_CUDA
        // synch to wait for kernels
        m_exec_conf->useContext();
        cudaThreadSynchronize();
        m_exec_conf->releaseContext();
#endif
        // set the flags
        *h_data = flags;
        }
    else
        {
        // set the flags
        *h_data = flags;
#ifdef ENABLE_CUDA
        // copy to the device
        m_exec_conf->useContext();
        cudaMemcpy(d_data, h_data, sizeof(T), cudaMemcpyHostToDevice);
        m_exec_conf->releaseContext();
#endif
        }
    }

template<class T> inline void GPUFlags<T>::resetFlagsThread(const T flags, unsigned int thread_id)
    {
    if (m_mapped)
        {
        // set the flags
        *h_data = flags;
        }
    else
        {
        // set the flags
        *h_data = flags;
#ifdef ENABLE_CUDA
        cudaStream_t stream = m_exec_conf->getThreadStream(thread_id);
        // copy to the device
        m_exec_conf->useContext();
        cudaMemcpyAsync(d_data, h_data, sizeof(T), cudaMemcpyHostToDevice,stream);
        m_exec_conf->releaseContext();
#endif
        }
    } 
#endif

