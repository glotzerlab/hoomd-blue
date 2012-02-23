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

/*! \file GPUArray.h
    \brief Defines the GPUArray class
*/

#ifndef __GPUARRAY_H__
#define __GPUARRAY_H__

// for vector types
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "ExecutionConfiguration.h"
#include <iostream>
#include <stdexcept>
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
        GPUArray(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
        //! Constructs a 2-D GPUArray
        GPUArray(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf);
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

        //! Resize the GPUArray
        /*! This method resizes the array by allocating a new array and copying over the elements
            from the old array. This is a slow process.
            Only data from the currently active memory location (gpu/cpu) is copied over to the resized
            memory area.
        */
        virtual void resize(unsigned int num_elements);

        //! Resize a 2D GPUArray
        virtual void resize(unsigned int width, unsigned int height);

    protected:
        //! Helper function to clear memory
        inline void memclear();

        //! Acquires the data pointer for use
        inline T* aquire(const access_location::Enum location, const access_mode::Enum mode, unsigned int gpu) const;

        //! Release the data pointer
        inline void release() const
            {
            m_acquired = false;
            }

    private:
        unsigned int m_num_elements;            //!< Number of elements
        unsigned int m_pitch;                   //!< Pitch of the rows in elements
        unsigned int m_height;                  //!< Number of allocated rows
        
        mutable bool m_acquired;                //!< Tracks whether the data has been aquired
        mutable data_location::Enum m_data_location;    //!< Tracks the current location of the data
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;    //!< execution configuration for working with CUDA
        
#ifdef ENABLE_CUDA
        mutable T* d_data;      //!< Pointer to allocated device memory
#endif
        
        mutable T* h_data;      //!< Pointer to allocated host memory
        
        //! Helper function to allocate memory
        inline void allocate();
        //! Helper function to free memory
        inline void deallocate();

        //! Helper function to copy memory from the device to host
        inline void memcpyDeviceToHost() const;
        //! Helper function to copy memory from the host to device
        inline void memcpyHostToDevice() const;

        //! Helper function to resize host array
        inline T* resizeHostArray(unsigned int num_elements);

        //! Helper function to resize a 2D host array
        inline T* resize2DHostArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height );

        //! Helper function to resize device array
        inline T* resizeDeviceArray(unsigned int num_elements);

        //! Helper function to resize a 2D device array
       inline T* resize2DDeviceArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height );

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
        m_num_elements(0), m_pitch(0), m_height(0), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    }

/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUArray<T>::GPUArray(unsigned int num_elements, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
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
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization and shutdown
*/
template<class T> GPUArray<T>::GPUArray(unsigned int width, unsigned int height, boost::shared_ptr<const ExecutionConfiguration> exec_conf) :
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

        
    // allocate memory
#ifdef ENABLE_CUDA
    assert(d_data == NULL);
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaHostAlloc(&h_data, m_num_elements*sizeof(T), cudaHostAllocDefault);
        cudaMalloc(&d_data, m_num_elements*sizeof(T));
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
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
        cudaFreeHost(h_data);
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
template<class T> void GPUArray<T>::memclear()
    {
    // don't do anything if there are no elements
    if (m_num_elements == 0)
        return;
        
    assert(h_data);
    
    // clear memory
    memset(h_data, 0, sizeof(T)*m_num_elements);
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
        cudaMemset(d_data, 0, m_num_elements*sizeof(T));
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
    cudaMemcpy(h_data, d_data, sizeof(T)*m_num_elements, cudaMemcpyDeviceToHost);
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
    cudaMemcpy(d_data, h_data, sizeof(T)*m_num_elements, cudaMemcpyHostToDevice);
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
template<class T> T* GPUArray<T>::aquire(const access_location::Enum location, const access_mode::Enum mode,
                                         unsigned int gpu) const
    {
    // sanity check
    assert(!m_acquired);
    m_acquired = true;
    
    // base case - handle acquiring a NULL GPUArray by simply returning NULL to prevent any memcpys from being attempted
    if (isNull())
        return NULL;
    
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
        if (!m_exec_conf || !m_exec_conf->isCUDAEnabled())
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

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template<class T> T* GPUArray<T>::resizeHostArray(unsigned int num_elements)
    {
    // if not allocated, do nothing
    if (isNull()) return NULL;

    // do not resize unless array is extended
    if (num_elements <= m_num_elements)
        return NULL;

    // allocate resized array
    T *h_tmp;
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaHostAlloc(&h_tmp, num_elements*sizeof(T), cudaHostAllocDefault);
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
    memcpy(h_tmp, h_data, sizeof(T)*m_num_elements);

    // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaFreeHost(h_data);
        }
    else
        {
        delete[] h_data;
        }
#else
    delete[] h_data;
#endif

    h_data = h_tmp;
    return h_data;
    }

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
*/
template<class T> T* GPUArray<T>::resize2DHostArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height )
    {
    // allocate resized array
    T *h_tmp;
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaHostAlloc(&h_tmp, new_pitch*new_height*sizeof(T), cudaHostAllocDefault);
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
    for (unsigned int i = 0; i < height; i++)
        memcpy(h_tmp + i * new_pitch, h_data + i*pitch, sizeof(T)*pitch);

    // free old memory location
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        cudaFreeHost(h_data);
        }
    else
        {
        delete[] h_data;
        }
#else
    delete[] h_data;
#endif

    h_data = h_tmp;
    return h_data;
    }

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
*/
template<class T> T* GPUArray<T>::resizeDeviceArray(unsigned int num_elements)
    {
#ifdef ENABLE_CUDA
    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, num_elements*sizeof(T));
    CHECK_CUDA_ERROR();

    assert(d_tmp);

    // clear memory
    cudaMemset(d_tmp, 0, num_elements*sizeof(T));
    CHECK_CUDA_ERROR();

    // copy over data
    cudaMemcpy(d_tmp, d_data, sizeof(T)*m_num_elements,cudaMemcpyDeviceToDevice);
    CHECK_CUDA_ERROR();

    // free old memory location
    cudaFree(d_data);
    CHECK_CUDA_ERROR();

    d_data = d_tmp;
    return d_data;
#else
    return NULL;
#endif
    }

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
*/
template<class T> T* GPUArray<T>::resize2DDeviceArray(unsigned int pitch, unsigned int new_pitch, unsigned int height, unsigned int new_height)
    {
#ifdef ENABLE_CUDA
    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, new_pitch*new_height*sizeof(T));
    CHECK_CUDA_ERROR();

    assert(d_tmp);

    // clear memory
    cudaMemset(d_tmp, 0, new_pitch*new_height*sizeof(T));
    CHECK_CUDA_ERROR();

    // copy over data
    // every column is copied separately such as to align with the new pitch
    for (unsigned int i = 0; i < height; i++)
        {
        cudaMemcpy(d_tmp + i * new_pitch, d_data + i * pitch, sizeof(T)*pitch,cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR();
        }

    // free old memory location
    cudaFree(d_data);
    CHECK_CUDA_ERROR();

    d_data = d_tmp;
    return d_data;
#else
    return NULL;
#endif
    }


/*! \param num_elements new size of array
*/
template<class T> void GPUArray<T>::resize(unsigned int num_elements)
    {
    assert(! m_acquired);
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
        {
        m_num_elements = num_elements;
        allocate();
        return;
        };

    // do not resize unless array is extended
    if (num_elements <= m_num_elements)
        return;

    resizeHostArray(num_elements);
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resizeDeviceArray(num_elements);
#endif
    m_num_elements = num_elements;
    }

/*! \param width new width of array
*   \param height new height of array
* \warning Resizing a 2D array can actually invalidate the stored data (as the pitch may change).
* It should not be expected that after a resize of a 2D array the contents are still accessible.
*/
template<class T> void GPUArray<T>::resize(unsigned int width, unsigned int height)
    {
    assert(! m_acquired);
    assert(num_elements > 0);

    // make m_pitch the next multiple of 16 larger or equal to the given width
    unsigned int new_pitch = (width + (16 - (width & 15)));

    unsigned int num_elements = new_pitch * height;

    // do not resize unless array is extended
    if (new_pitch <= new_pitch && height <= m_height)
        return;

    // it is allowed to resize only one dimension, then the other dimension
    // is forced to stay the same
    if (m_pitch > new_pitch)
        new_pitch = m_pitch;
    if (m_height > new_pitch)
        height = m_height;

    // if not allocated, simply allocate
    if (isNull())
        {
        m_num_elements = num_elements;
        allocate();
        m_pitch = new_pitch;
        m_height = height;
        return;
        };

    resize2DHostArray(m_pitch, new_pitch, m_height, height);
#ifdef ENABLE_CUDA
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resize2DDeviceArray(m_pitch, new_pitch, m_height, height);
#endif
    m_num_elements = num_elements;

    m_height = height;
    m_pitch  = new_pitch;
    m_num_elements = m_pitch * m_height;
    }
#endif

