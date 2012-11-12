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

/*! \file GPUBuffer.h
    \brief Defines the GPUBuffer class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __GPUBUFFER_H__
#define __GPUBUFFER_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "ExecutionConfiguration.h"

#include <iostream>
#include <stdexcept>

//! Class for managing a buffer to copy data between the host and the GPU
/*! GPUBuffer is a class that manages a mapped pinned memory buffer.
    It provides two methods to access the data: getHostPointer() returns a pointer to the host buffer,
    and getDevicePointer() returns the pointer to the device memory. Data written to the host data
    is automatically available on the device, and data written to the device buffer is seen on the host
    after a call to getHostPointer().

    The class is used for data that needs to be frequently exchanged between the GPU and the host.

    Similar to GPUArray, it supports methods for resizing as well as 2D buffers.

    \ingroup data_structs
*/
template<class T> class GPUBuffer
    {
    public:
        //! Constructs a NULL GPUBuffer
        GPUBuffer();
        //! Constructs a GPUBuffer attached to a GPU
        GPUBuffer(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Constructs a two-dimensional GPUBuffer 
        GPUBuffer(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Frees memory
        ~GPUBuffer();
        
        //! Copy constructor
        GPUBuffer(const GPUBuffer& from);
        //! = operator
        GPUBuffer& operator=(const GPUBuffer& rhs);
        
        //! Swap the pointers in two GPUBuffer
        inline void swap(GPUBuffer& from);
        
        //! Test if the GPUBuffer is NULL
        bool isNull() const
            {
            return (h_data == NULL);
            }
            
        //! Get a pointer to the data on the host
        inline T* getHostPointer();

#ifdef ENABLE_CUDA
        //! Get a pointer to the data on the device
        T* getDevicePointer()
            {
            return d_data;
            }

        //! Get a pointer to the data on the device (const version)
        const T* getDevicePointer() const
            {
            return d_data;
            }
#endif

        //! Get the number of elements
        /*!
         - For 1-D allocated GPUBuffers, this is the number of elements allocated.
         - For 2-D allocated GPUBuffers, this is the \b total number of elements (\a pitch * \a height) allocated
        */
        unsigned int getNumElements() const
            {
            return m_num_elements;
            }
 
        //! Get the width of the allocated rows in elements
        /*!
         - For 2-D allocated GPUBuffers, this is the total width of a row in memory (including the padding added for coalescing)
         - For 1-D allocated GPUBuffers, this is the simply the number of elements allocated.
        */
        unsigned int getPitch() const
            {
            return m_pitch;
            }
            
        //! Get the number of rows allocated
        /*!
         - For 2-D allocated GPUBuffers, this is the height given to the constructor
         - For 1-D allocated GPUBuffers, this is the simply 1.
        */
        unsigned int getHeight() const
            {
            return m_height;
            }

        //! Resize the GPUBuffer
        /*! This method resizes the array by allocating a new array and copying over the elements
            from the old array. This is a slow process.
        */
        virtual void resize(unsigned int num_elements);

        //! Resize a 2D GPUBuffer
        virtual void resize(unsigned int width, unsigned int height);

    
    private:
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;    //!< execution configuration for working with CUDA
        unsigned int m_num_elements; //!< Number of elements in buffer
        unsigned int m_height;       //!< Height of 2D buffer
        unsigned int m_pitch;        //!< Pitch of 2D buffer
        
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

        //! Helper function to resize host array
        inline T* resizeBuffer(unsigned int num_elements);

        //! Helper function to resize a 2D host array
        inline T* resize2DBuffer(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height );
 
    };

//******************************************
// GPUBuffer implementation
// *****************************************

template<class T> GPUBuffer<T>::GPUBuffer() :
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    }

/*! \param width Number of columns
    \param height Number of rows
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUBuffer<T>::GPUBuffer(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
        m_exec_conf(exec_conf), m_height(height),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
#ifdef ENABLE_CUDA 
    // we require mapped pinned memory
    if (m_exec_conf->isCUDAEnabled() && !m_exec_conf->dev_prop.canMapHostMemory) 
        {
        m_exec_conf->msg->error() << "Device does not support mapped pinned memory." << std::endl << std::endl;
        throw std::runtime_error("Error initializing GPUBuffer.");
        }
#endif 

    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));
    
    // setup the number of elements
    m_num_elements = m_pitch * m_height;
 
    // allocate and clear memory
    allocate();
    memclear();
    }

/*! \param ments Number of elements to allocate
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUBuffer<T>::GPUBuffer(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
        m_exec_conf(exec_conf), m_num_elements(num_elements), m_height(1), m_pitch(num_elements),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
#ifdef ENABLE_CUDA 
    // we require mapped pinned memory
    if (m_exec_conf->isCUDAEnabled() && !m_exec_conf->dev_prop.canMapHostMemory) 
        {
        m_exec_conf->msg->error() << "Device does not support mapped pinned memory." << std::endl << std::endl;
        throw std::runtime_error("Error initializing GPUBuffer.");
        }
#endif 

    // allocate and clear memory
    allocate();
    memclear();
    }



template<class T> GPUBuffer<T>::~GPUBuffer()
    {
    deallocate();
    }

template<class T> GPUBuffer<T>::GPUBuffer(const GPUBuffer& from) : m_exec_conf(from.m_exec_conf),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();
  
#ifdef ENABLE_CUDA
    // synchronize
    if (m_exec_conf->isCUDAEnabled())
        cudaDeviceSynchronize();
#endif

    // copy over the data to the new GPUBuffer
    memcpy(h_data, from.h_data, from.m_num_elements);
    }

template<class T> GPUBuffer<T>& GPUBuffer<T>::operator=(const GPUBuffer& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // free current memory
        deallocate();
        
        // copy over basic elements
        m_exec_conf = rhs.m_exec_conf;
        m_num_elements = rhs.m_num_elements;
        m_height = rhs.m_height;
        m_pitch = rhs.m_pitch;
        
        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();

#ifdef ENABLE_CUDA
        if (m_exec_conf->isCUDAEnabled())
            // synchronize
            cudaDeviceSynchronize();
#endif

        // copy over the data to the new GPUBuffer
        memcpy(h_data, rhs.h_data, rhs.m_num_elements);
        }
        
    return *this;
    }

/*! \param from GPUBuffer to swap \a this with

    a.swap(b) will result in the equivalent of:
    \code
GPUBuffer c(a);
a = b;
b = c;
    \endcode

    But it will be done in a super-efficent way by just swapping the internal pointers, thus avoiding all the expensive
    memory deallocations/allocations and copies using the copy constructor and assignment operator.
*/
template<class T> void GPUBuffer<T>::swap(GPUBuffer& from)
    {
    std::swap(m_num_elements, from.m_num_elements);
    std::swap(m_height, from.m_height);
    std::swap(m_pitch, from.m_pitch);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
#endif
    std::swap(h_data, from.h_data);
    }

/*! \pre pointers are not allocated
    \post All memory pointers needed for GPUBuffer are allocated
*/
template<class T> void GPUBuffer<T>::allocate()
    {
    // sanity check
    assert(h_data == NULL);

    // allocate memory
#ifdef ENABLE_CUDA
    assert(d_data == NULL);
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        void *ptr = NULL;
        posix_memalign(&ptr, getpagesize(), sizeof(T)*m_num_elements);
        h_data = (T *) ptr;
 
        cudaHostRegister(h_data, sizeof(T)*m_num_elements, cudaHostRegisterMapped);
        cudaHostGetDevicePointer(&d_data, h_data, 0);
        CHECK_CUDA_ERROR();
        }
    else
        {
        h_data = new T[m_num_elements];
        }
#else
    h_data = new T[m_num_elements];
#endif
    }

/*! \post All allocated memory is freed
*/
template<class T> void GPUBuffer<T>::deallocate()
    {
    // don't do anything if the pointers have not been allocated
    if (h_data == NULL)
        return;
        
    // free memory
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();
        free(h_data);
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
template<class T> void GPUBuffer<T>::memclear()
    {
    // don't do anything if nothing is allocated
    if (h_data == NULL)
        return;
        
    assert(h_data);
    
#ifdef ENABLE_CUDA
    // wait for the device to catch up
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaDeviceSynchronize();
        }
#endif

    // clear memory
    memset(h_data, 0, sizeof(T)*m_num_elements);
    }

/*! \returns Pointer to the host buffer
    \note getHostPointer implicitly synchronizes with the GPU execution stream.
    If there are any previous asynch kernel
    launches that may set the flags, getHostPointer() will wait until they complete and will the current data
    possibly set by those kernels.
*/
template<class T> T* GPUBuffer<T>::getHostPointer()
    {
#ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // synch to wait for kernels
        cudaDeviceSynchronize();
        }
#endif    

    // return pointer to data
    return h_data;
    }

/*! \post Memory is resized, the newly allocated part of the array is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template<class T> T* GPUBuffer<T>::resizeBuffer(unsigned int num_elements)
    {
    // if not allocated, do nothing
    if (isNull()) return NULL;

    // allocate resized array
    T *h_tmp = NULL;
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        void *ptr = NULL;
        posix_memalign(&ptr, getpagesize(), num_elements*sizeof(T));
        cudaHostRegister(ptr, num_elements*sizeof(T), cudaHostRegisterMapped);
        CHECK_CUDA_ERROR();
        h_tmp = (T *) ptr;
        }
    else
        {
        h_tmp = new T[num_elements];
        }
#else
    h_tmp = new T[num_elements];
#endif

    // clear memory
    memset(h_tmp, 0, sizeof(T)*num_elements);

    // copy over data
    unsigned int num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;
    memcpy(h_tmp, h_data, sizeof(T)*num_copy_elements);

    // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();
        free(h_data);
        }
    else
        {
        delete[] h_data;
        }
#else
    delete[] h_data;
#endif

    h_data = h_tmp;

    cudaHostGetDevicePointer(&d_data, h_data, 0);

    return h_data;
    }

/*! \post Memory is resized, the newly allocated part of the array is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template<class T> T* GPUBuffer<T>::resize2DBuffer(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height )
    {
    // allocate resized array
    T *h_tmp = NULL;
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        unsigned int size = new_pitch*new_height*sizeof(T);
        void *ptr = NULL;
        posix_memalign(&ptr, getpagesize(), size);
        cudaHostRegister(ptr, size, cudaHostRegisterMapped);
        CHECK_CUDA_ERROR();
        h_tmp = (T *) ptr;
        }
    else
        {
        h_tmp = new T[new_pitch * new_height];
        }
#else
    h_tmp = new T[new_pitch * new_height];
#endif

    // clear memory
    memset(h_tmp, 0, sizeof(T)*new_pitch*new_height);

    // copy over data
    // every column is copied separately such as to align with the new pitch
    unsigned int num_copy_rows = height > new_height ? new_height : height;
    unsigned int num_copy_columns = pitch > new_pitch ? new_pitch : pitch;
    for (unsigned int i = 0; i < num_copy_rows; i++)
        memcpy(h_tmp + i * new_pitch, h_data + i*pitch, sizeof(T)*num_copy_columns);

    // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaHostUnregister(h_data);
        CHECK_CUDA_ERROR();
        free(h_data);
        }
    else
        {
        delete[] h_data;
        }
#else
    delete[] h_data;
#endif

    h_data = h_tmp;

    cudaHostGetDevicePointer(&d_data, h_data, 0);

    return h_data;
    }

/*! \param num_elements new size of array
*/
template<class T> void GPUBuffer<T>::resize(unsigned int num_elements)
    {
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
        {
        m_num_elements = num_elements;
        allocate();
        return;
        };

    m_exec_conf->msg->notice(7) << "GPUBuffer: Resizing to " << float(num_elements*sizeof(T))/1024.0f/1024.0f << " MB" << std::endl;

    resizeBuffer(num_elements);

    m_num_elements = num_elements;
    m_pitch = num_elements;
    }

/*! \param width new width of array
*   \param height new height of array
*/
template<class T> void GPUBuffer<T>::resize(unsigned int width, unsigned int height)
    {

    // make m_pitch the next multiple of 16 larger or equal to the given width
    unsigned int new_pitch = (width + (16 - (width & 15)));

    unsigned int num_elements = new_pitch * height;
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
        {
        m_num_elements = num_elements;
        allocate();
        m_pitch = new_pitch;
        m_height = height;
        return;
        };

    m_exec_conf->msg->notice(7) << "GPUBuffer: Resizing to " << float(num_elements*sizeof(T))/1024.0f/1024.0f << " MB" << std::endl;

    resize2DBuffer(m_pitch, new_pitch, m_height, height);
    m_num_elements = num_elements;

    m_height = height;
    m_pitch  = new_pitch;
    m_num_elements = m_pitch * m_height;
    }

#endif

