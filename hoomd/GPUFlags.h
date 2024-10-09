// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file GPUFlags.h
    \brief Defines the GPUFlags class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __GPUFLAGS_H__
#define __GPUFLAGS_H__

// for vector types
#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "ExecutionConfiguration.h"

#include <iostream>
#include <stdexcept>

namespace hoomd
    {
//! Class for managing a small set of flags set on the GPU
/*!
\b Overview:

GPUFlags is an efficient container for a small set of flags. It is optimized for use when a given
set of flags is set via a kernel call on the GPU and then read on the host. The host may also reset
the flag to a known value.

\b Implementation details:
Like GPUArray, GPUFlags keeps around an ExecutionConfiguration to keep the GPU alive, and the flags
are "mirrored" on the host and device. In order to optimize for the fastest performance of the host
reading flags that were set on the GPU, the flags are stored in host mapped memory (on devices that
support it). GPUFlags inserts synchronization points 1) when resetFlags() is called to reset the
value of the flags (otherwise a running kernel may overwrite the reset value) and 2) when reading
the flags on the host with readFlags(). These synchronizations cost about 5-10 microseconds, so do
not read or reset the flags more times than is needed.

\ingroup data_structs
*/
template<class T> class PYBIND11_EXPORT GPUFlags
    {
    public:
    //! Constructs a NULL GPUFlags
    GPUFlags();
    //! Constructs a GPUFlags attached to a GPU
    GPUFlags(std::shared_ptr<const ExecutionConfiguration> exec_conf);
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

#ifdef ENABLE_HIP
    //! Get the flags on the device
    T* getDeviceFlags()
        {
        return d_data;
        }
#endif

    private:
    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< execution configuration for working with CUDA
    bool m_mapped;   //!< Set to true when using host mapped memory

#ifdef ENABLE_HIP
    mutable T* d_data; //!< Pointer to allocated device memory
#endif
    mutable T* h_data; //!< Pointer to allocated host memory

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

template<class T>
GPUFlags<T>::GPUFlags()
    : m_mapped(false),
#ifdef ENABLE_HIP
      d_data(NULL),
#endif
      h_data(NULL)
    {
    }

/*! \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization
 * and shutdown
 */
template<class T>
GPUFlags<T>::GPUFlags(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_exec_conf(exec_conf), m_mapped(false),
#ifdef ENABLE_HIP
      d_data(NULL),
#endif
      h_data(NULL)
    {
#ifdef ENABLE_HIP
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

template<class T>
GPUFlags<T>::GPUFlags(const GPUFlags& from)
    : m_exec_conf(from.m_exec_conf), m_mapped(false),
#ifdef ENABLE_HIP
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

    But it will be done in a super-efficient way by just swapping the internal pointers, thus
avoiding all the expensive memory deallocations/allocations and copies using the copy constructor
and assignment operator.
*/
template<class T> void GPUFlags<T>::swap(GPUFlags& from)
    {
    std::swap(m_mapped, from.m_mapped);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_HIP
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
#ifdef ENABLE_HIP
    assert(d_data == NULL);
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        if (m_mapped)
            {
#ifdef ENABLE_MPI
            void* ptr = NULL;
            // need to use hooks provided by MPI library
            int retval = posix_memalign(&ptr, getpagesize(), sizeof(T));
            if (retval != 0)
                {
                throw std::runtime_error("Error allocating aligned memory.");
                }
            h_data = (T*)ptr;
            hipHostRegister(h_data, sizeof(T), hipHostRegisterMapped);
#else
            hipHostMalloc(&h_data, sizeof(T), hipHostMallocMapped);
#endif
            CHECK_CUDA_ERROR();
            hipHostGetDevicePointer((void**)&d_data, h_data, 0);
            CHECK_CUDA_ERROR();
            }
        else
            {
#ifdef ENABLE_MPI
            void* ptr = NULL;
            int retval = posix_memalign(&ptr, getpagesize(), sizeof(T));
            if (retval != 0)
                {
                throw std::runtime_error("Error allocating aligned memory.");
                }
            h_data = (T*)ptr;
            hipHostRegister(h_data, sizeof(T), hipHostRegisterDefault);
#else
            hipHostMalloc(&h_data, sizeof(T), hipHostMallocDefault);
#endif
            CHECK_CUDA_ERROR();
            hipMalloc(&d_data, sizeof(T));
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
#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
#ifdef ENABLE_MPI
        hipHostUnregister(h_data);
        free(h_data);
#else
        hipHostFree(h_data);
#endif
        if (!m_mapped)
            {
            hipFree(d_data);
            }
        }
    else
        {
        delete h_data;
        }
#else
    delete h_data;
#endif

    // set pointers to NULL
    h_data = NULL;
#ifdef ENABLE_HIP
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

#ifdef ENABLE_HIP
    // wait for the device to catch up
    if (m_exec_conf && m_exec_conf->isCUDAEnabled() && m_mapped)
        {
        hipDeviceSynchronize();
        }
#endif

    // clear memory
    memset(h_data, 0, sizeof(T));
#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled() && !m_mapped)
        {
        assert(d_data);
        hipMemset(d_data, 0, sizeof(T));
        }
#endif
    }

/*! \returns Current value of the flags
    \note readFlags implicitly synchronizes with the GPU execution stream. If there are any previous
   asynch kernel launches that may set the flags, readFlags() will wait until they complete and will
   return any flags possibly set by those kernels.
*/
template<class T> const T GPUFlags<T>::readFlags()
    {
#ifdef ENABLE_HIP
    if (m_mapped)
        {
        // synch to wait for kernels
        hipDeviceSynchronize();
        }
    else
        {
        if (m_exec_conf->isCUDAEnabled())
            {
            // memcpy the results to the host
            hipMemcpy(h_data, d_data, sizeof(T), hipMemcpyDeviceToHost);
            }
        }
#endif

    // return value of flags
    return *h_data;
    }

/*! \param flags Value of flags to set
    \note resetFlags synchronizes with the GPU execution stream. It waits for all prior kernel
   launches to complete before actually resetting the flags so that a possibly executing kernel
   doesn't unintentionally overwrite the intended reset value.
*/
template<class T> void GPUFlags<T>::resetFlags(const T flags)
    {
    if (m_mapped)
        {
#ifdef ENABLE_HIP
        // synch to wait for kernels
        hipDeviceSynchronize();
#endif
        // set the flags
        *h_data = flags;
        }
    else
        {
        // set the flags
        *h_data = flags;
#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            // copy to the device
            hipMemcpy(d_data, h_data, sizeof(T), hipMemcpyHostToDevice);
            }
#endif
        }
    }

    } // end namespace hoomd

#endif
