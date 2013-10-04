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

// Maintainer: jglaser

/*! \file GPUBufferMapped.h
    \brief Defines the GPUBufferMapped class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// this class is only available for GPU use
#ifdef ENABLE_CUDA

#ifndef __GPU_BUFFER_MAPPED_H__
#define __GPU_BUFFER_MAPPED_H__

#include "GPUArray.h"

#include <iostream>
#include <stdexcept>

//! Class for managing a buffer to copy data between the host and the GPU
/*! GPUBufferMapped is a class that manages a mapped pinned memory buffer.
    It provides two methods to access the data: getHostPointer() returns a pointer to the host buffer,
    and getDevicePointer() returns the pointer to the device memory. Data written to the host data
    is automatically available on the device, and data written to the device buffer is seen on the host
    after a call to getHostPointer().

    The class is used for data that needs to be frequently exchanged between the GPU and the host.

    Similar to GPUArray, it supports methods for resizing as well as 2D buffers.

    \ingroup data_structs
*/
template<class T> class GPUBufferMapped : public GPUArray<T>
    {
    public:
        //! Constructs a NULL GPUBufferMapped
        GPUBufferMapped();
        //! Constructs a GPUBufferMapped attached to a GPU
        GPUBufferMapped(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Constructs a two-dimensional GPUBufferMapped
        GPUBufferMapped(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Frees memory
        ~GPUBufferMapped() { }

        //! Get a pointer to the data on the host
        inline T* getHostPointer();

        //! Get a pointer to the data on the host (const version)
        inline const T* getHostPointer() const;

#ifdef ENABLE_CUDA
        //! Get a pointer to the data on the device
        T* getDevicePointer()
            {
            return this->d_data;
            }

        //! Get a pointer to the data on the device (const version)
        const T* getDevicePointer() const
            {
            return this->d_data;
            }
#endif

    };

//******************************************
// GPUBufferMapped implementation
//*****************************************

//! empty constructor
template<class T> GPUBufferMapped<T>::GPUBufferMapped()
    {
    }

/*! \param width Number of columns
    \param height Number of rows
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUBufferMapped<T>::GPUBufferMapped(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
    GPUArray<T>(width, height, exec_conf, true)
    {
    }

/*! \param num_elements Number of elements to allocate
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUBufferMapped<T>::GPUBufferMapped(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
        GPUArray<T>(num_elements, exec_conf, true)
    {
    }

template<class T> T* GPUBufferMapped<T>::getHostPointer()
    {
#ifdef ENABLE_CUDA
    if (this->m_exec_conf->isCUDAEnabled())
        {
        // synch to wait for kernels
        cudaDeviceSynchronize();
        }
#endif

    // return pointer to data
    return this->h_data;
    }

/*! \returns Pointer to the host buffer (const version)
 */
template<class T> const T* GPUBufferMapped<T>::getHostPointer() const
    {
#ifdef ENABLE_CUDA
    if (this->m_exec_conf->isCUDAEnabled())
        {
        // synch to wait for kernels
        cudaDeviceSynchronize();
        }
#endif

    // return pointer to data
    return this->h_data;
    }

#endif  // __GPU_BUFFER_MAPPED_H__
#endif  // ENABLE_CUDA
