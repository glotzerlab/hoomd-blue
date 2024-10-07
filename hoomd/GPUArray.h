// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file GPUArray.h
    \brief Defines the GPUArray class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#pragma once

// 4 GB is considered a large allocation for a single GPU buffer, and user should be warned
#define LARGEALLOCBYTES 0xffffffff

// for vector types
#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "ExecutionConfiguration.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>

#include <cxxabi.h>
#include <sstream>

namespace hoomd
    {
//! Specifies where to acquire the data
struct access_location
    {
    //! The enum
    enum Enum
        {
        host, //!< Ask to acquire the data on the host
#ifdef ENABLE_HIP
        device //!< Ask to acquire the data on the device
#endif
        };
    };

//! Defines where the data is currently stored
struct data_location
    {
    //! The enum
    enum Enum
        {
        host, //!< Data was last updated on the host
#ifdef ENABLE_HIP
        device,    //!< Data was last updated on the device
        hostdevice //!< Data is up to date on both the host and device
#endif
        };
    };

//! Specify how the data is to be accessed
struct access_mode
    {
    //! The enum
    enum Enum
        {
        read,      //!< Data will be accessed read only
        readwrite, //!< Data will be accessed for read and write
        overwrite  //!< The data is to be completely overwritten during this acquire
        };
    };

template<class T> class GPUArray;

namespace detail
    {
template<class T> class device_deleter
    {
    public:
    //! Default constructor
    device_deleter() : m_use_device(false), m_N(0), m_mapped(false) { }

    //! Ctor
    /*! \param exec_conf Execution configuration
        \param use_device whether the array is managed or on the host
     */
    device_deleter(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                   bool use_device,
                   const size_t N,
                   bool mapped)
        : m_exec_conf(exec_conf), m_use_device(use_device), m_N(N), m_mapped(mapped)
        {
        }

    //! Delete the host array
    /*! \param ptr Start of aligned memory allocation
     */
    void operator()(T* ptr)
        {
        if (ptr == nullptr)
            return;

        if (m_use_device && !m_mapped)
            {
            assert(m_exec_conf);
            this->m_exec_conf->msg->notice(10)
                << "Freeing " << m_N * sizeof(T) << " bytes of CUDA memory." << std::endl;

#ifdef ENABLE_HIP
            hipFree(ptr);
#endif
            }
        }

    private:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
    bool m_use_device;                                         //!< Whether to use cudaMallocManaged
    size_t m_N;                                                //!< Number of elements in array
    bool m_mapped; //!< True if this is host-mapped memory
    };

template<class T> class host_deleter
    {
    public:
    //! Default constructor
    host_deleter() : m_use_device(false), m_N(0) { }

    //! Ctor
    /*! \param exec_conf Execution configuration
        \param use_device whether the array is managed or on the host
     */
    host_deleter(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                 bool use_device,
                 const size_t N)
        : m_exec_conf(exec_conf), m_use_device(use_device), m_N(N)
        {
        }

    //! Delete the CUDA array
    /*! \param ptr Start of aligned memory allocation
     */
    void operator()(T* ptr)
        {
        if (ptr == nullptr)
            return;

        if (m_exec_conf)
            m_exec_conf->msg->notice(10)
                << "Freeing " << m_N * sizeof(T) << " bytes of host memory." << std::endl;

        if (m_use_device)
            {
            assert(m_exec_conf);

// unregister host memory from CUDA driver
#if (ENABLE_HIP)
            hipHostUnregister(ptr);
#endif
            }

        // free the allocation
        free(ptr);
        }

    private:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
    bool m_use_device;                                         //!< Whether to use hostMallocManaged
    size_t m_N;                                                //!< Number of elements in array
    };
    } // end namespace detail

//! Forward declarations
template<class T> class ArrayHandleDispatch;

template<class T> class GPUArrayDispatch;

template<class T> class ArrayHandle;

template<class T> class ArrayHandleAsync;

template<class T> class GlobalArray;

//! CRTP (Curiously recurring template pattern) interface for GPUArray/GlobalArray
template<class T, class Derived> class GPUArrayBase
    {
    public:
    //! Get the number of elements
    /*!
     - For 1-D allocated GPUArrays, this is the number of elements allocated.
     - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height)
     allocated
    */
    size_t getNumElements() const
        {
        return static_cast<Derived const&>(*this).getNumElements();
        }

    //! Test if the GPUArray is NULL
    bool isNull() const
        {
        return static_cast<Derived const&>(*this).isNull();
        }

    //! Get the width of the allocated rows in elements
    /*!
     - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the
     padding added for coalescing)
     - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
    */
    size_t getPitch() const
        {
        return static_cast<Derived const&>(*this).getPitch();
        }

    //! Get the number of rows allocated
    /*!
     - For 2-D allocated GPUArrays, this is the height given to the constructor
     - For 1-D allocated GPUArrays, this is the simply 1.
    */
    size_t getHeight() const
        {
        return static_cast<Derived const&>(*this).getHeight();
        }

    //! Resize the GPUArray
    void resize(size_t num_elements)
        {
        static_cast<Derived&>(*this).resize(num_elements);
        }

    //! Resize a 2D GPUArray
    void resize(size_t width, size_t height)
        {
        static_cast<Derived&>(*this).resize(width, height);
        }

    protected:
    //! Acquires the data pointer for use
    inline ArrayHandleDispatch<T> acquire(const access_location::Enum location,
                                          const access_mode::Enum mode
#ifdef ENABLE_HIP
                                          ,
                                          bool async = false
#endif
    ) const
        {
        return static_cast<Derived const&>(*this).acquire(location,
                                                          mode
#ifdef ENABLE_HIP
                                                          ,
                                                          async
#endif
        );
        }

    //! Release the data pointer
    inline void release() const
        {
        return static_cast<Derived const&>(*this).release();
        }

    //! Returns the acquire state
    inline bool isAcquired() const
        {
        return static_cast<Derived const&>(*this).isAcquired();
        }

    // need to be friend of the ArrayHandle class
    friend class ArrayHandle<T>;
    friend class ArrayHandleAsync<T>;

    private:
    // Make constructor private to prevent mistakes
    GPUArrayBase() { };
    friend Derived;
    };

//! This base class is the glue between the ArrayHandle and a generic GPUArrayBase<Derived>
template<class T> class ArrayHandleDispatch
    {
    public:
    //! Constructor
    ArrayHandleDispatch(T* const _data) : data(_data) { }

    //! Get the data pointer
    T* const get() const
        {
        return data;
        }

    //! Destructor
    virtual ~ArrayHandleDispatch() = default;

    private:
    //! The data pointer
    T* const data;
    };

//! Handle to access the data pointer handled by GPUArray
/*! The data in GPUArray is only accessible via ArrayHandle. The pointer is accessible for the
lifetime of the ArrayHandle. When the ArrayHandle is destroyed, the GPUArray is notified that the
data has been released. This tracking mechanism provides for error checking that will cause code
assertions to fail if the data is acquired more than once.

    ArrayHandle is intended to be used within a scope limiting its use. For example:
    \code
GPUArray<int> gpu_array(100);

    {
    ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
    ... use h_handle.data ...
    }
    \endcode

    The actual raw pointer \a data should \b NOT be assumed to be the same after the handle is
released. The pointer may in fact be re-allocated somewhere else after the handle is released and
before the next handle is acquired.

    \ingroup data_structs
*/
template<class T> class ArrayHandle
    {
    public:
    //! Aquires the data and sets \a data
    /*! \tparam Derived the type of GPUArray implementation
     */
    template<class Derived>
    inline ArrayHandle(const GPUArrayBase<T, Derived>& gpu_array,
                       const access_location::Enum location = access_location::host,
                       const access_mode::Enum mode = access_mode::readwrite);
    //! Notifies the containing GPUArray that the handle has been released
    virtual inline ~ArrayHandle() = default;

    private:
    ArrayHandleDispatch<T>
        dispatch; //!< Reference to the dispatch object that manages the acquire/release

    public:
    T* const data; //!< Pointer to data
    };

#ifdef ENABLE_HIP
//! Implementation of ArrayHandle using asynchronous copying between host and device
/*! This handle can be used to speed up access to the GPUArray data when
    accessing multiple buffers on the host AND the device.

    ArrayHandleAsync is asynchronous with respect to the host, i.e. multiple
    ArrayHandleAsync objects maybe instantiated for multiple GPUArrays in a row, without
    incurring significant overhead for each of the handles.

    \warning Because ArrayHandleAsync uses asynchronous copying, however, array data is not
    guaranteed to be available on the host unless the device has been synchronized.

    Example usage:
    \code
GPUArray<int> gpu_array_1(100);
GPUArray<int> gpu_array_2(100);

    {
    ArrayHandle<int> h_handle_1(gpu_array_1, access_location::host, access_mode::readwrite);
    ArrayHandle<int> h_handle_2(gpu_array_2, access_location:::host, access_mode::readwrite);
    cudaDeviceSynchronize();

    ... use h_handle_1.data and h_handle_2.data ...
    }
    \endcode
*/
template<class T> class ArrayHandleAsync
    {
    public:
    //! Aquires the data and sets \a data using asynchronous copies
    /*! \tparam Derived the type of GPUArray implementation
     */
    template<class Derived>
    inline ArrayHandleAsync(const GPUArrayBase<T, Derived>& gpu_array,
                            const access_location::Enum location = access_location::host,
                            const access_mode::Enum mode = access_mode::readwrite);

    //! Notifies the containing GPUArray that the handle has been released
    virtual inline ~ArrayHandleAsync() = default;

    private:
    ArrayHandleDispatch<T> dispatch; //!< Dispatch object that manages the acquire/release

    public:
    T* const data; //!< Pointer to data
    };
#endif

//*******
//! Class for managing an array of elements on the GPU mirrored to the CPU
/*!
GPUArray provides a template class for managing the majority of the GPU<->CPU memory usage patterns
in HOOMD. It represents a single array of elements which is present both on the CPU and GPU. Via
ArrayHandle, classes can access the array pointers through a handle for a short time. All needed
memory transfers from the host <-> device are handled by the class based on the access mode and
location specified when acquiring an ArrayHandle.

GPUArray is fairly advanced, C++ wise. It is a template class, so GPUArray's of floats, float4's,
uint2's, etc.. can be made. It comes with a copy constructor and = operator so you can (expensively)
pass GPUArray's around in arguments or overwrite one with another via assignment (inexpensive swaps
can be performed with swap()). The ArrayHandle acquisition method guarantees that every acquired
handle will be released. About the only thing it \b doesn't do is prevent the user from writing to a
pointer acquired with a read only mode.

At a high level, GPUArray encapsulates a smart pointer \a std::unique_ptr<T> \a data and with \a
num_elements elements, and keeps a copy of this data on both the host and device. When accessing
this data through the construction of an ArrayHandle instance, the \a location (host or device) you
wish to access the data must be specified along with an access \a mode (read, readwrite, overwrite).

When the data is accessed in the same location it was last written to, the pointer is simply
returned. If the data is accessed in a different location, it will be copied before the pointer is
returned.

When the data is accessed in the \a read mode, it is assumed that the data will not be written to
and thus there is no need to copy memory the next time the data is acquired somewhere else. Using
the readwrite mode specifies that the data is to be read and written to, necessitating possible
copies to the desired location before the data can be accessed and again before the next access. If
the data is to be completely overwritten \b without reading it first, then an expensive memory copy
can be avoided by using the \a overwrite mode.

Data with both 1-D and 2-D representations can be allocated by using the appropriate constructor.
2-D allocated data is still just a flat pointer, but the row width is rounded up to a multiple of
16 elements to facilitate coalescing. The actual allocated width is accessible with getPitch(). Here
is an example of addressing element i,j in a 2-D allocated GPUArray.
\code
GPUArray<int> gpu_array(100, 200, m_exec_conf);
size_t pitch = gpu_array.getPitch();

ArrayHandle<int> h_handle(gpu_array, access_location::host, access_mode::readwrite);
h_handle.data[i*pitch + j] = 5;
\endcode

A future modification of GPUArray will allow mirroring or splitting the data across multiple GPUs.

\ingroup data_structs
*/
template<class T> class GPUArray : public GPUArrayBase<T, GPUArray<T>>
    {
    public:
    //! Constructs a NULL GPUArray
    GPUArray();
    //! Constructs a NULL GPUArray with an execution configuration
    GPUArray(std::shared_ptr<const ExecutionConfiguration> exec_conf);

    //! Constructs a 1-D GPUArray
    GPUArray(size_t num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf);
    //! Constructs a 2-D GPUArray
    GPUArray(size_t width, size_t height, std::shared_ptr<const ExecutionConfiguration> exec_conf);
    //! Frees memory
    virtual ~GPUArray() { }

#ifdef ENABLE_HIP
    //! Constructs a 1-D GPUArray
    GPUArray(size_t num_elements,
             std::shared_ptr<const ExecutionConfiguration> exec_conf,
             bool mapped);
    //! Constructs a 2-D GPUArray
    GPUArray(size_t width,
             size_t height,
             std::shared_ptr<const ExecutionConfiguration> exec_conf,
             bool mapped);
#endif

    //! Copy constructor
    GPUArray(const GPUArray& from) noexcept;
    //! = operator
    GPUArray& operator=(const GPUArray& rhs) noexcept;

    //! Move constructor
    GPUArray(GPUArray&& from) noexcept;
    //! Move assignment operator
    GPUArray& operator=(GPUArray&& rhs) noexcept;

    //! Swap the pointers in two GPUArrays
    inline void swap(GPUArray& from);

    //! Get the number of elements
    /*!
     - For 1-D allocated GPUArrays, this is the number of elements allocated.
     - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height)
     allocated
    */
    size_t getNumElements() const
        {
        return m_num_elements;
        }

    //! Test if the GPUArray is NULL
    bool isNull() const
        {
        return !h_data;
        }

    //! Get the width of the allocated rows in elements
    /*!
     - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the
     padding added for coalescing)
     - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
    */
    size_t getPitch() const
        {
        return m_pitch;
        }

    //! Get the number of rows allocated
    /*!
     - For 2-D allocated GPUArrays, this is the height given to the constructor
     - For 1-D allocated GPUArrays, this is the simply 1.
    */
    size_t getHeight() const
        {
        return m_height;
        }

    //! Resize the GPUArray
    /*! This method resizes the array by allocating a new array and copying over the elements
        from the old array. This is a slow process.
        Only data from the currently active memory location (gpu/cpu) is copied over to the resized
        memory area.
    */
    void resize(size_t num_elements);

    //! Resize a 2D GPUArray
    void resize(size_t width, size_t height);

    //! Return a string representation of this array
    std::string getRepresentation() const
        {
        if (!isNull())
            {
            std::ostringstream o;
            const std::string type_name = typeid(T).name();
            int status;
            char* realname = abi::__cxa_demangle(type_name.c_str(), 0, 0, &status);
            if (status)
                throw std::runtime_error("Status " + std::to_string(status)
                                         + " while trying to demangle data type.");

            o << h_data.get() << "-" << h_data.get() + m_num_elements;

#ifdef ENABLE_HIP
            if (m_exec_conf->isCUDAEnabled())
                o << " (host) " << d_data.get() << "-" << d_data.get() + m_num_elements
                  << " (device)";
#endif

            o << " [" << realname << "]";
            free(realname);
            return o.str();
            }
        else
            return std::string("null");
        }

    //! get the execution configuration
    std::shared_ptr<const ExecutionConfiguration> getExecutionConfiguration()
        {
        return m_exec_conf;
        }

    protected:
    //! Clear memory starting from a given element
    /*! \param first The first element to clear
     */
    inline void memclear(size_t first = 0);

    //! Acquires the data pointer for use
    inline ArrayHandleDispatch<T> acquire(const access_location::Enum location,
                                          const access_mode::Enum mode
#ifdef ENABLE_HIP
                                          ,
                                          bool async = false
#endif
    ) const;

    //! Release the data pointer
    inline void release() const
        {
        m_acquired = false;
        }

    //! Returns the acquire state
    inline bool isAcquired() const
        {
        return m_acquired;
        }

    //! Need to be friend with dispatch
    friend class ArrayHandleDispatch<T>;
    friend class GPUArrayDispatch<T>;

    friend class GPUArrayBase<T, GPUArray<T>>;

    // GlobalArray is our friend, too, to enable fall back
    friend class GlobalArray<T>;

    private:
    size_t m_num_elements; //!< Number of elements
    size_t m_pitch;        //!< Pitch of the rows in elements
    size_t m_height;       //!< Number of allocated rows

    mutable bool m_acquired;                     //!< Tracks whether the data has been acquired
    mutable data_location::Enum m_data_location; //!< Tracks the current location of the data
#ifdef ENABLE_HIP
    bool m_mapped; //!< True if we are using mapped memory
#endif

    // ok, this looks weird, but I want m_exec_conf to be protected and not have to go reorder all
    // of the initializers
    protected:
#ifdef ENABLE_HIP
    std::unique_ptr<T, hoomd::detail::device_deleter<T>>
        d_data; //!< Smart pointer to allocated device memory
#endif

    std::unique_ptr<T, hoomd::detail::host_deleter<T>> h_data; //!< Pointer to allocated host memory

    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< execution configuration for working with CUDA

    private:
    //! Helper function to allocate memory
    inline void allocate();

#ifdef ENABLE_HIP
    //! Helper function to copy memory from the device to host
    inline void memcpyDeviceToHost(bool async) const;
    //! Helper function to copy memory from the host to device
    inline void memcpyHostToDevice(bool async) const;
#endif

    //! Helper function to resize host array
    inline T* resizeHostArray(size_t num_elements);

    //! Helper function to resize a 2D host array
    inline T* resize2DHostArray(size_t pitch, size_t new_pitch, size_t height, size_t new_height);

    //! Helper function to resize device array
    inline T* resizeDeviceArray(size_t num_elements);

    //! Helper function to resize a 2D device array
    inline T* resize2DDeviceArray(size_t pitch, size_t new_pitch, size_t height, size_t new_height);
    };

//******************************************
// ArrayHandle implementation
// *****************************************

/*! \param gpu_array GPUArray host to the pointer data
    \param location Desired location to access the data
    \param mode Mode to access the data with
*/
template<class T>
template<class Derived>
ArrayHandle<T>::ArrayHandle(const GPUArrayBase<T, Derived>& array,
                            const access_location::Enum location,
                            const access_mode::Enum mode)
    : dispatch(array.acquire(location, mode)), data(dispatch.get())
    {
    }

#ifdef ENABLE_HIP
template<class T>
template<class Derived>
ArrayHandleAsync<T>::ArrayHandleAsync(const GPUArrayBase<T, Derived>& array,
                                      const access_location::Enum location,
                                      const access_mode::Enum mode)
    : dispatch(array.acquire(location, mode, true)), data(dispatch.get())
    {
    }
#endif

//************************************************
// ArrayHandleDispatch specialization for GPUArray
// ***********************************************

template<class T> class GPUArrayDispatch : public ArrayHandleDispatch<T>
    {
    public:
    GPUArrayDispatch(T* const _data, const GPUArray<T>& _gpu_array)
        : ArrayHandleDispatch<T>(_data), gpu_array(_gpu_array)
        {
        }

    virtual ~GPUArrayDispatch()
        {
        assert(gpu_array.isAcquired());
        gpu_array.release();
        }

    private:
    const GPUArray<T>& gpu_array;
    };

//******************************************
// GPUArray implementation
// *****************************************

template<class T>
GPUArray<T>::GPUArray()
    : m_num_elements(0), m_pitch(0), m_height(0), m_acquired(false),
      m_data_location(data_location::host)
#ifdef ENABLE_HIP
      ,
      m_mapped(false)
#endif
    {
    }

template<class T>
GPUArray<T>::GPUArray(std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_num_elements(0), m_pitch(0), m_height(0), m_acquired(false),
      m_data_location(data_location::host),
#ifdef ENABLE_HIP
      m_mapped(false),
#endif
      m_exec_conf(exec_conf)
    {
    }

/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization
   and shutdown
*/
template<class T>
GPUArray<T>::GPUArray(size_t num_elements, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false),
      m_data_location(data_location::host),
#ifdef ENABLE_HIP
      m_mapped(false),
#endif
      m_exec_conf(exec_conf)
    {
    // allocate and clear memory
    allocate();
    memclear();
    }

/*! \param width Width of the 2-D array to allocate (in elements)
    \param height Number of rows to allocate in the 2D array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization
   and shutdown
*/
template<class T>
GPUArray<T>::GPUArray(size_t width,
                      size_t height,
                      std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_height(height), m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_HIP
      m_mapped(false),
#endif
      m_exec_conf(exec_conf)
    {
    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));

    // setup the number of elements
    m_num_elements = m_pitch * m_height;

    // allocate and clear memory
    allocate();
    memclear();
    }

#ifdef ENABLE_HIP
/*! \param num_elements Number of elements to allocate in the array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization
   and shutdown \param mapped True if we are using mapped-pinned memory
*/
template<class T>
GPUArray<T>::GPUArray(size_t num_elements,
                      std::shared_ptr<const ExecutionConfiguration> exec_conf,
                      bool mapped)
    : m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false),
      m_data_location(data_location::host), m_mapped(mapped), m_exec_conf(exec_conf)
    {
    // allocate and clear memory
    allocate();
    memclear();
    }

/*! \param width Width of the 2-D array to allocate (in elements)
    \param height Number of rows to allocate in the 2D array
    \param exec_conf Shared pointer to the execution configuration for managing CUDA initialization
   and shutdown \param mapped True if we are using mapped-pinned memory
*/
template<class T>
GPUArray<T>::GPUArray(size_t width,
                      size_t height,
                      std::shared_ptr<const ExecutionConfiguration> exec_conf,
                      bool mapped)
    : m_height(height), m_acquired(false), m_data_location(data_location::host), m_mapped(mapped),
      m_exec_conf(exec_conf)
    {
    // make m_pitch the next multiple of 16 larger or equal to the given width
    m_pitch = (width + (16 - (width & 15)));

    // setup the number of elements
    m_num_elements = m_pitch * m_height;

    // allocate and clear memory
    allocate();
    memclear();
    }
#endif

template<class T>
GPUArray<T>::GPUArray(const GPUArray& from) noexcept
    : m_num_elements(from.m_num_elements), m_pitch(from.m_pitch), m_height(from.m_height),
      m_acquired(false), m_data_location(data_location::host),
#ifdef ENABLE_HIP
      m_mapped(from.m_mapped),
#endif
      m_exec_conf(from.m_exec_conf)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();

    // copy over the data to the new GPUArray
    if (from.h_data.get())
        {
        ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
        memcpy(h_data.get(), h_handle.data, sizeof(T) * m_num_elements);
        }
    }

template<class T> GPUArray<T>& GPUArray<T>::operator=(const GPUArray& rhs) noexcept
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // sanity check
        assert(!m_acquired && !rhs.m_acquired);

        // copy over basic elements
        m_num_elements = rhs.m_num_elements;
        m_pitch = rhs.m_pitch;
        m_height = rhs.m_height;
        m_exec_conf = rhs.m_exec_conf;
#ifdef ENABLE_HIP
        m_mapped = rhs.m_mapped;
#endif
        // initialize state variables
        m_data_location = data_location::host;

        // copy over the data to the new GPUArray
        if (rhs.h_data)
            {
            // allocate and clear new memory the same size as the data in rhs
            allocate();
            memclear();

            ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
            memcpy(h_data.get(), h_handle.data, sizeof(T) * m_num_elements);
            }
        else
            {
            h_data.reset();

#ifdef ENABLE_HIP
            d_data.reset();
#endif
            }
        }

    return *this;
    }

//! Move C'tor
template<class T>
GPUArray<T>::GPUArray(GPUArray&& from) noexcept
    : m_num_elements(std::move(from.m_num_elements)), m_pitch(std::move(from.m_pitch)),
      m_height(std::move(from.m_height)), m_acquired(std::move(from.m_acquired)),
      m_data_location(std::move(from.m_data_location)),
#ifdef ENABLE_HIP
      m_mapped(std::move(from.m_mapped)), d_data(std::move(from.d_data)),
#endif
      h_data(std::move(from.h_data)), m_exec_conf(std::move(from.m_exec_conf))
    {
    }

//! Move assignment operator
template<class T> GPUArray<T>& GPUArray<T>::operator=(GPUArray&& rhs) noexcept
    {
    if (&rhs != this)
        {
        m_num_elements = std::move(rhs.m_num_elements);
        m_pitch = std::move(rhs.m_pitch);
        m_height = std::move(rhs.m_height);
        m_exec_conf = std::move(rhs.m_exec_conf);
#ifdef ENABLE_HIP
        m_mapped = std::move(rhs.m_mapped);
        d_data = std::move(rhs.d_data);
#endif
        h_data = std::move(rhs.h_data);
        m_data_location = std::move(rhs.m_data_location);
        m_acquired = std::move(rhs.m_acquired);
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

    But it will be done in a super-efficient way by just swapping the internal pointers, thus
avoiding all the expensive memory deallocations/allocations and copies using the copy constructor
and assignment operator.
*/
template<class T> void GPUArray<T>::swap(GPUArray& from)
    {
    // this may work, but really shouldn't be done when acquired
    assert(!m_acquired && !from.m_acquired);
    assert(&from != this);

    std::swap(m_num_elements, from.m_num_elements);
    std::swap(m_pitch, from.m_pitch);
    std::swap(m_height, from.m_height);
    std::swap(m_acquired, from.m_acquired);
    std::swap(m_data_location, from.m_data_location);
    std::swap(m_exec_conf, from.m_exec_conf);
#ifdef ENABLE_HIP
    std::swap(d_data, from.d_data);
    std::swap(m_mapped, from.m_mapped);
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

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (size_t)sizeof(T) && m_exec_conf)
        {
        m_exec_conf->msg->notice(7)
            << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
        }

#ifdef ENABLE_HIP
    // we require mapped pinned memory
    if (m_mapped && m_exec_conf && !m_exec_conf->dev_prop.canMapHostMemory)
        {
        if (m_exec_conf)
            m_exec_conf->msg->error()
                << "Device does not support mapped pinned memory." << std::endl
                << std::endl;
        throw std::runtime_error("Error allocating GPUArray.");
        }
#endif

    if (m_exec_conf)
        m_exec_conf->msg->notice(10)
            << "GPUArray: Allocating " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f
            << " MB" << std::endl;

    void* host_ptr = nullptr;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign(&host_ptr, 32, m_num_elements * sizeof(T));
    if (retval != 0)
        {
        throw std::bad_alloc();
        }

    bool use_device = m_exec_conf && m_exec_conf->isCUDAEnabled();

#ifdef ENABLE_HIP
    void* device_ptr = nullptr;
    if (use_device)
        {
// register pointer for DMA
#ifdef ENABLE_HIP
        hipHostRegister(host_ptr,
                        m_num_elements * sizeof(T),
                        m_mapped ? hipHostRegisterMapped : hipHostRegisterDefault);
#endif
        CHECK_CUDA_ERROR();
        }
#endif

    // store in smart ptr with custom deleter
    hoomd::detail::host_deleter<T> host_deleter(m_exec_conf, use_device, m_num_elements);
    h_data = std::unique_ptr<T, hoomd::detail::host_deleter<T>>(reinterpret_cast<T*>(host_ptr),
                                                                host_deleter);

#if defined(ENABLE_HIP)
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        // Check for pending errors.
        CHECK_CUDA_ERROR();

        // allocate and/or map host memory
        if (m_mapped)
            {
#ifdef ENABLE_HIP
            hipError_t error = hipHostGetDevicePointer(&device_ptr, h_data.get(), 0);
            if (error == hipErrorMemoryAllocation)
                {
                throw std::bad_alloc();
                }
            else if (error != hipSuccess)
                {
                throw std::runtime_error(hipGetErrorString(error));
                }
#endif
            }
        else
            {
#ifdef ENABLE_HIP
            hipError_t error = hipMalloc(&device_ptr, m_num_elements * sizeof(T));
            if (error == hipErrorMemoryAllocation)
                {
                throw std::bad_alloc();
                }
            else if (error != hipSuccess)
                {
                throw std::runtime_error(hipGetErrorString(error));
                }
#endif
            }

        // store in smart pointer with custom deleter
        hoomd::detail::device_deleter<T> device_deleter(m_exec_conf,
                                                        use_device,
                                                        m_num_elements,
                                                        m_mapped);
        d_data
            = std::unique_ptr<T, hoomd::detail::device_deleter<T>>(reinterpret_cast<T*>(device_ptr),
                                                                   device_deleter);
        }
#endif
    }

/*! \pre allocate() has been called
    \post All allocated memory is set to 0
*/
template<class T> void GPUArray<T>::memclear(size_t first)
    {
    // don't do anything if there are no elements
    if (!h_data.get())
        return;

    assert(h_data);
    assert(first < m_num_elements);

    // clear memory
    memset((void*)(h_data.get() + first), 0, sizeof(T) * (m_num_elements - first));

#if defined(ENABLE_HIP)
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
        assert(d_data);
        if (!m_mapped)
#ifdef ENABLE_HIP
            hipMemset(d_data.get() + first, 0, (m_num_elements - first) * sizeof(T));
#endif
        }
#endif
    }

#if defined(ENABLE_HIP)
/*! \post All memory on the device is copied to the host array
 */
template<class T> void GPUArray<T>::memcpyDeviceToHost(bool async) const
    {
    // don't do anything if there are no elements
    if (!h_data.get())
        return;

    if (m_mapped)
        {
// if we are using mapped pinned memory, no need to copy, only synchronize
#ifdef ENABLE_HIP
        if (!async)
            hipDeviceSynchronize();
#endif
        return;
        }

    if (m_exec_conf)
        m_exec_conf->msg->notice(10)
            << "GPUArray: Copying " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f
            << " MB device->host " << (async ? std::string("async") : std::string()) << std::endl;
#ifdef ENABLE_HIP
    if (async)
        {
        hipMemcpyAsync(h_data.get(),
                       d_data.get(),
                       sizeof(T) * m_num_elements,
                       hipMemcpyDeviceToHost);
        }
    else
        {
        hipMemcpy(h_data.get(), d_data.get(), sizeof(T) * m_num_elements, hipMemcpyDeviceToHost);
        }
#endif
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

/*! \post All memory on the host is copied to the device array
 */
template<class T> void GPUArray<T>::memcpyHostToDevice(bool async) const
    {
    // don't do anything if there are no elements
    if (!h_data.get())
        return;

    if (m_mapped)
        {
        // if we are using mapped pinned memory, no need to copy
        // rely on CUDA's implicit synchronization
        return;
        }

    if (m_exec_conf)
        m_exec_conf->msg->notice(10)
            << "GPUArray: Copying " << float(m_num_elements * sizeof(T)) / 1024.0f / 1024.0f
            << " MB host->device " << (async ? std::string("async") : std::string()) << std::endl;
    if (async)
#ifdef ENABLE_HIP
        hipMemcpyAsync(d_data.get(),
                       h_data.get(),
                       sizeof(T) * m_num_elements,
                       hipMemcpyHostToDevice);
#endif
    else
#ifdef ENABLE_HIP
        hipMemcpy(d_data.get(), h_data.get(), sizeof(T) * m_num_elements, hipMemcpyHostToDevice);
#endif
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }
#endif

/*! \param location Desired location to access the data
    \param mode Mode to access the data with
    \param async True if array copying should be done async

    acquire() is the workhorse of GPUArray. It tracks the internal state variable \a data_location
   and performs all host<->device memory copies as needed during the state changes given the
    specified access mode and location where the data is to be acquired.

    acquire() cannot be directly called by the user class. Data must be accessed through
   ArrayHandle.
*/
template<class T>
ArrayHandleDispatch<T> GPUArray<T>::acquire(const access_location::Enum location,
                                            const access_mode::Enum mode
#ifdef ENABLE_HIP
                                            ,
                                            bool async
#endif
) const
    {
    if (m_acquired)
        {
        throw std::runtime_error("Cannot acquire access to array in use.");
        }
    m_acquired = true;

    // base case - handle acquiring a NULL GPUArray by simply returning NULL to prevent any memcpys
    // from being attempted
    if (isNull())
        return GPUArrayDispatch<T>(nullptr, *this);

    // first, break down based on where the data is to be acquired
    if (location == access_location::host)
        {
        // then break down based on the current location of the data
        if (m_data_location == data_location::host)
            {
            // the state stays on the host regardles of the access mode
            return GPUArrayDispatch<T>(h_data.get(), *this);
            }
#ifdef ENABLE_HIP
        else if (m_data_location == data_location::hostdevice)
            {
            // finally perform the action based on the access mode requested
            if (mode == access_mode::read) // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite) // state goes to host
                m_data_location = data_location::host;
            else if (mode == access_mode::overwrite) // state goes to host
                m_data_location = data_location::host;
            else
                {
                throw std::runtime_error("Invalid access mode requested.");
                }

            return GPUArrayDispatch<T>(h_data.get(), *this);
            }
        else if (m_data_location == data_location::device)
            {
            // finally perform the action based on the access mode requested
            if (mode == access_mode::read)
                {
                // need to copy data from the device to the host
                memcpyDeviceToHost(async);
                // state goes to hostdevice
                m_data_location = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                // need to copy data from the device to the host
                memcpyDeviceToHost(async);
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
                throw std::runtime_error("Invalid access mode requested.");
                }

            return GPUArrayDispatch<T>(h_data.get(), *this);
            }
#endif
        else
            {
            throw std::runtime_error("Invalid data location state.");
            return ArrayHandleDispatch<T>(nullptr);
            }
        }
#ifdef ENABLE_HIP
    else if (location == access_location::device)
        {
        // check that a GPU is actually specified
        if (!m_exec_conf)
            {
            throw std::runtime_error(
                "Requesting device acquire, but we have no execution configuration");
            }
        if (!m_exec_conf->isCUDAEnabled())
            {
            m_exec_conf->msg->error()
                << "Requesting device acquire, but no GPU in the Execution Configuration"
                << std::endl;
            throw std::runtime_error("Error acquiring data");
            }

        // then break down based on the current location of the data
        if (m_data_location == data_location::host)
            {
            // finally perform the action based on the access mode requested
            if (mode == access_mode::read)
                {
                // need to copy data to the device
                memcpyHostToDevice(async);
                // state goes to hostdevice
                m_data_location = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                // need to copy data to the device
                memcpyHostToDevice(async);
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
                throw std::runtime_error("Invalid access mode requested.");
                }

            return GPUArrayDispatch<T>(d_data.get(), *this);
            }
        else if (m_data_location == data_location::hostdevice)
            {
            // finally perform the action based on the access mode requested
            if (mode == access_mode::read) // state stays on hostdevice
                m_data_location = data_location::hostdevice;
            else if (mode == access_mode::readwrite) // state goes to device
                m_data_location = data_location::device;
            else if (mode == access_mode::overwrite) // state goes to device
                m_data_location = data_location::device;
            else
                {
                throw std::runtime_error("Invalid access mode requested.");
                }
            return GPUArrayDispatch<T>(d_data.get(), *this);
            }
        else if (m_data_location == data_location::device)
            {
            // the stat stays on the device regardless of the access mode
            return GPUArrayDispatch<T>(d_data.get(), *this);
            }
        else
            {
            throw std::runtime_error("Invalid data_location state.");
            return ArrayHandleDispatch<T>(nullptr);
            }
        }
#endif
    else
        {
        throw std::runtime_error("Invalid location requested.");
        return ArrayHandleDispatch<T>(nullptr);
        }
    }

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
 */
template<class T> T* GPUArray<T>::resizeHostArray(size_t num_elements)
    {
    // if not allocated, do nothing
    if (isNull())
        return NULL;

    // allocate resized array
    T* h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void**)&h_tmp, 32, num_elements * sizeof(T));
    if (retval != 0)
        {
        throw std::bad_alloc();
        }

#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
#ifdef ENABLE_HIP
        hipHostRegister(h_tmp,
                        num_elements * sizeof(T),
                        m_mapped ? hipHostRegisterMapped : hipHostRegisterDefault);
#endif
        CHECK_CUDA_ERROR();
        }
#endif
    // clear memory
    memset((void*)h_tmp, 0, sizeof(T) * num_elements);

    // copy over data
    size_t num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;
    memcpy((void*)h_tmp, (void*)h_data.get(), sizeof(T) * num_copy_elements);

    // update smart pointer
    bool use_device = m_exec_conf && m_exec_conf->isCUDAEnabled();
    hoomd::detail::host_deleter<T> host_deleter(m_exec_conf, use_device, num_elements);
    h_data = std::unique_ptr<T, hoomd::detail::host_deleter<T>>(h_tmp, host_deleter);

#ifdef ENABLE_HIP
    // update device pointer
    if (m_mapped)
        {
        void* dev_ptr = nullptr;
#ifdef ENABLE_HIP
        hipHostGetDevicePointer(&dev_ptr, h_data.get(), 0);
#endif

        // no-op deleter
        hoomd::detail::device_deleter<T> device_deleter(m_exec_conf,
                                                        use_device,
                                                        num_elements,
                                                        true);
        d_data = std::unique_ptr<T, hoomd::detail::device_deleter<T>>(reinterpret_cast<T*>(dev_ptr),
                                                                      device_deleter);
        }
#endif

    return h_data.get();
    }

/*! \post Memory on the host is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a pointer to the newly allocated memory area
 */
template<class T>
T* GPUArray<T>::resize2DHostArray(size_t pitch, size_t new_pitch, size_t height, size_t new_height)
    {
    // allocate resized array
    T* h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    size_t size = new_pitch * new_height * sizeof(T);
    int retval = posix_memalign((void**)&h_tmp, 32, size);
    if (retval != 0)
        {
        throw std::bad_alloc();
        }

#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        {
#ifdef ENABLE_HIP
        hipHostRegister(h_tmp, size, m_mapped ? hipHostRegisterMapped : hipHostRegisterDefault);
#endif
        CHECK_CUDA_ERROR();
        }
#endif

    // clear memory
    memset((void*)h_tmp, 0, sizeof(T) * new_pitch * new_height);

    // copy over data
    // every column is copied separately such as to align with the new pitch
    size_t num_copy_rows = height > new_height ? new_height : height;
    size_t num_copy_columns = pitch > new_pitch ? new_pitch : pitch;
    for (size_t i = 0; i < num_copy_rows; i++)
        memcpy((void*)(h_tmp + i * new_pitch),
               (void*)(h_data.get() + i * pitch),
               sizeof(T) * num_copy_columns);

    // update smart pointer
    bool use_device = m_exec_conf && m_exec_conf->isCUDAEnabled();
    hoomd::detail::host_deleter<T> host_deleter(m_exec_conf, use_device, new_pitch * new_height);
    h_data = std::unique_ptr<T, hoomd::detail::host_deleter<T>>(h_tmp, host_deleter);

#ifdef ENABLE_HIP
    // update device pointer
    if (m_mapped)
        {
        void* dev_ptr = nullptr;
#ifdef ENABLE_HIP
        hipHostGetDevicePointer(&dev_ptr, h_data.get(), 0);
#endif

        // no-op deleter
        hoomd::detail::device_deleter<T> device_deleter(m_exec_conf,
                                                        use_device,
                                                        new_pitch * new_height,
                                                        true);
        d_data = std::unique_ptr<T, hoomd::detail::device_deleter<T>>(reinterpret_cast<T*>(dev_ptr),
                                                                      device_deleter);
        }

#endif

    return h_data.get();
    }

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
 */
template<class T> T* GPUArray<T>::resizeDeviceArray(size_t num_elements)
    {
    // Check for pending errors.
    CHECK_CUDA_ERROR();

#ifdef ENABLE_HIP
    if (m_mapped)
        return NULL;

    // allocate resized array
    T* d_tmp;
#ifdef ENABLE_HIP
    hipError_t error = hipMalloc(&d_tmp, num_elements * sizeof(T));
    if (error == hipErrorMemoryAllocation)
        {
        throw std::bad_alloc();
        }
    else if (error != hipSuccess)
        {
        throw std::runtime_error(hipGetErrorString(error));
        }
#endif

    assert(d_tmp);

// clear memory
#ifdef ENABLE_HIP
    hipMemset(d_tmp, 0, num_elements * sizeof(T));
#endif
    CHECK_CUDA_ERROR();

    // copy over data
    size_t num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;
#ifdef ENABLE_HIP
    hipMemcpy(d_tmp, d_data.get(), sizeof(T) * num_copy_elements, hipMemcpyDeviceToDevice);
#endif
    CHECK_CUDA_ERROR();

    // update smart ptr
    hoomd::detail::device_deleter<T> device_deleter(m_exec_conf,
                                                    m_exec_conf->isCUDAEnabled(),
                                                    num_elements,
                                                    m_mapped);
    d_data = std::unique_ptr<T, hoomd::detail::device_deleter<T>>(d_tmp, device_deleter);

    return d_data.get();
#else
    return NULL;
#endif
    }

/*! \post Memory on the device is resized, the newly allocated part of the array
 *        is reset to zero
 *! \returns a device pointer to the newly allocated memory area
 */
template<class T>
T* GPUArray<T>::resize2DDeviceArray(size_t pitch,
                                    size_t new_pitch,
                                    size_t height,
                                    size_t new_height)
    {
#ifdef ENABLE_HIP
    // Check for pending errors.
    CHECK_CUDA_ERROR();

    if (m_mapped)
        return NULL;

    // allocate resized array
    T* d_tmp;
#ifdef ENABLE_HIP
    hipError_t error = hipMalloc(&d_tmp, new_pitch * new_height * sizeof(T));
    if (error == hipErrorMemoryAllocation)
        {
        throw std::bad_alloc();
        }
    else if (error != hipSuccess)
        {
        throw std::runtime_error(hipGetErrorString(error));
        }
#endif
    assert(d_tmp);

// clear memory
#ifdef ENABLE_HIP
    hipMemset(d_tmp, 0, new_pitch * new_height * sizeof(T));
#endif
    CHECK_CUDA_ERROR();

    // copy over data
    // every column is copied separately such as to align with the new pitch
    size_t num_copy_rows = height > new_height ? new_height : height;
    size_t num_copy_columns = pitch > new_pitch ? new_pitch : pitch;

    for (size_t i = 0; i < num_copy_rows; i++)
        {
#ifdef ENABLE_HIP
        hipMemcpy(d_tmp + i * new_pitch,
                  d_data.get() + i * pitch,
                  sizeof(T) * num_copy_columns,
                  hipMemcpyDeviceToDevice);
#endif
        CHECK_CUDA_ERROR();
        }

    // update smart ptr
    hoomd::detail::device_deleter<T> device_deleter(m_exec_conf,
                                                    m_exec_conf->isCUDAEnabled(),
                                                    new_pitch * new_height,
                                                    m_mapped);
    d_data = std::unique_ptr<T, hoomd::detail::device_deleter<T>>(d_tmp, device_deleter);

    return d_data.get();
#else
    return NULL;
#endif
    }

/*! \param num_elements new size of array
 *
 * \warning An array can be expanded or shrunk, depending on the parameters supplied.
 *          It is the responsibility of the caller to ensure that no data is inadvertently lost when
 *          reducing the size of the array.
 */
template<class T> void GPUArray<T>::resize(size_t num_elements)
    {
    assert(!m_acquired);
    assert(num_elements > 0);

    // if not allocated, simply allocate
    if (isNull())
        {
        m_num_elements = num_elements;
        allocate();
        return;
        };

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (size_t)sizeof(T) && m_exec_conf)
        {
        m_exec_conf->msg->notice(7)
            << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
        }

    if (m_exec_conf)
        m_exec_conf->msg->notice(7)
            << "GPUArray: Resizing to " << float(num_elements * sizeof(T)) / 1024.0f / 1024.0f
            << " MB" << std::endl;

    resizeHostArray(num_elements);
#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resizeDeviceArray(num_elements);
#endif
    m_num_elements = num_elements;
    m_pitch = num_elements;
    }

/*! \param width new width of array
 *   \param height new height of array
 *
 *   \warning An array can be expanded or shrunk, depending on the parameters supplied.
 *   It is the responsibility of the caller to ensure that no data is inadvertently lost when
 *   reducing the size of the array.
 */
template<class T> void GPUArray<T>::resize(size_t width, size_t height)
    {
    assert(!m_acquired);

    // make m_pitch the next multiple of 16 larger or equal to the given width
    size_t new_pitch = (width + (16 - (width & 15)));

    size_t num_elements = new_pitch * height;
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

    // notify at a high level if a large allocation is about to occur
    if (m_num_elements > LARGEALLOCBYTES / (size_t)sizeof(T) && m_exec_conf)
        {
        m_exec_conf->msg->notice(7)
            << "GPUArray is trying to allocate a very large (>4GB) amount of memory." << std::endl;
        }

    resize2DHostArray(m_pitch, new_pitch, m_height, height);
#ifdef ENABLE_HIP
    if (m_exec_conf && m_exec_conf->isCUDAEnabled())
        resize2DDeviceArray(m_pitch, new_pitch, m_height, height);
#endif
    m_num_elements = num_elements;

    m_height = height;
    m_pitch = new_pitch;
    m_num_elements = m_pitch * m_height;
    }

    } // end namespace hoomd
