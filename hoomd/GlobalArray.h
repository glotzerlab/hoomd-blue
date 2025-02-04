// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file GlobalArray.h
    \brief Defines the GlobalArray class
*/

/*! GlobalArray internally uses managed memory to store data, to allow buffers being accessed from
    multiple devices.

    hipMemAdvise() can be called on GlobalArray's data, which is obtained using ::get().

    GlobalArray<> supports all functionality that GPUArray<> does, and should eventually replace
   GPUArray. In fact, for performance considerations in single GPU situations, GlobalArray
   internally falls back on GPUArray (and whenever it doesn't have an ExecutionConfiguration). This
   behavior is controlled by the result of ExecutionConfiguration::allConcurrentManagedAccess().

    One difference to GPUArray is that GlobalArray doesn't zero its memory space, so it is important
   to explicitly initialize the data.

    Internally, GlobalArray<> uses a smart pointer to comply with RAII semantics.

    As for GPUArray, access to the data is provided through ArrayHandle<> objects, with proper
   access_mode and access_location flags.
*/

#pragma once

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory>

#include "GPUArray.h"

#include <cxxabi.h>
#include <utility>

#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#define TAG_ALLOCATION(array)              \
        {                                  \
        array.setTag(std::string(#array)); \
        }

namespace hoomd
    {
namespace detail
    {
#ifdef __GNUC__
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
/* Test for GCC < 5.0 */
#if GCC_VERSION < 50000
    // work around GCC missing feature

#define NO_STD_ALIGN
// https://stackoverflow.com/questions/27064791/stdalign-not-supported-by-g4-9
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
inline void* my_align(std::size_t alignment, std::size_t size, void*& ptr, std::size_t& space)
    {
    std::uintptr_t pn = reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t aligned = (pn + alignment - 1) & -alignment;
    std::size_t padding = aligned - pn;
    if (space < size + padding)
        return nullptr;
    space -= padding;
    return ptr = reinterpret_cast<void*>(aligned);
    }
#endif
#endif

template<class T> class managed_deleter
    {
    public:
    //! Default constructor
    managed_deleter()
        : m_use_device(false), m_N(0), m_allocation_ptr(nullptr), m_allocation_bytes(0)
        {
        }

    //! Ctor
    /*! \param exec_conf Execution configuration
        \param use_device whether the array is managed or on the host
        \param N number of elements
        \param allocation_ptr true start of allocation, before alignment
        \param allocation_bytes Size of allocation
     */
    managed_deleter(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                    bool use_device,
                    std::size_t N,
                    void* allocation_ptr,
                    size_t allocation_bytes)
        : m_exec_conf(exec_conf), m_use_device(use_device), m_N(N),
          m_allocation_ptr(allocation_ptr), m_allocation_bytes(allocation_bytes)
        {
        }

    //! Set the tag
    void setTag(const std::string& tag)
        {
        m_tag = tag;
        }

    //! Destroy the items and delete the managed array
    /*! \param ptr Start of aligned memory allocation
     */
    void operator()(T* ptr)
        {
        if (ptr == nullptr)
            return;

        if (!m_exec_conf)
            return;

#ifdef ENABLE_HIP
        if (m_use_device)
            {
            hipDeviceSynchronize();
            CHECK_CUDA_ERROR();
            }
#endif

        // we used placement new in the allocation, so call destructors explicitly
        for (std::size_t i = 0; i < m_N; ++i)
            {
            ptr[i].~T();
            }

#ifdef ENABLE_HIP
        if (m_use_device)
            {
            std::ostringstream oss;
            oss << "Freeing " << m_allocation_bytes << " bytes of managed memory";
            if (m_tag != "")
                oss << " [" << m_tag << "]";
            oss << std::endl;
            this->m_exec_conf->msg->notice(10) << oss.str();

            hipFree(m_allocation_ptr);
            }
        else
#endif
            {
            free(m_allocation_ptr);
            }
        }

    std::pair<const void*, const void*> getAllocationRange() const
        {
        return std::make_pair(m_allocation_ptr,
                              reinterpret_cast<char*>(m_allocation_ptr)
                                  + sizeof(T) * m_allocation_bytes);
        }

    private:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
    bool m_use_device;                                         //!< Whether to use hipMallocManaged
    size_t m_N;                                                //!< Number of elements in array
    void* m_allocation_ptr;                                    //!< Start of unaligned allocation
    size_t m_allocation_bytes;                                 //!< Size of actual allocation
    std::string m_tag;                                         //!< Name of the array
    };

#ifdef ENABLE_HIP
class event_deleter
    {
    public:
    //! Default constructor
    event_deleter() { }

    //! Constructor with execution configuration
    /*! \param exec_conf The execution configuration (needed for CHECK_CUDA_ERROR)
     */
    event_deleter(std::shared_ptr<const ExecutionConfiguration> exec_conf) : m_exec_conf(exec_conf)
        {
        }

    //! Destroy the event and free the memory location
    /*! \param ptr Start of memory area
     */
    void operator()(hipEvent_t* ptr)
        {
        hipEventDestroy(*ptr);
        CHECK_CUDA_ERROR();

        delete ptr;
        }

    private:
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
    };
#endif

    } // end namespace detail

//! Forward declarations
template<class T> class GlobalArrayDispatch;

template<class T> class ArrayHandle;

//! Definition of GlobalArray using CRTP
template<class T> class GlobalArray : public GPUArrayBase<T, GlobalArray<T>>
    {
    public:
    //! Empty constructor
    GlobalArray()
        : m_num_elements(0), m_pitch(0), m_height(0), m_acquired(false), m_align_bytes(0),
          m_is_managed(false)
        {
        }

    /*! Allocate a 1D array in managed memory
        \param num_elements Number of elements in array
        \param exec_conf The current execution configuration
     */
    GlobalArray(size_t num_elements,
                std::shared_ptr<const ExecutionConfiguration> exec_conf,
                const std::string& tag = std::string(),
                bool force_managed = false)
        : m_exec_conf(exec_conf),
#ifndef ALWAYS_USE_MANAGED_MEMORY
          // explicit copy should be elided
          m_fallback((exec_conf->allConcurrentManagedAccess()
                      || (force_managed && exec_conf->isCUDAEnabled()))
                         ? GPUArray<T>()
                         : GPUArray<T>(num_elements, exec_conf)),
#endif
          m_num_elements(num_elements), m_pitch(num_elements), m_height(1), m_acquired(false),
          m_tag(tag), m_align_bytes(0),
          m_is_managed(exec_conf->allConcurrentManagedAccess()
                       || (force_managed && exec_conf->isCUDAEnabled()))
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!(m_is_managed))
            return;
#endif

        assert(this->m_exec_conf);
#ifdef ENABLE_HIP
        if (this->m_exec_conf->isCUDAEnabled())
            {
            // use OS page size as minimum alignment
            m_align_bytes = getpagesize();
            }
#endif

        if (m_num_elements > 0)
            allocate();
        }

    //! Destructor
    virtual ~GlobalArray() { }

    //! Copy constructor
    GlobalArray(const GlobalArray& from) noexcept
        : m_exec_conf(from.m_exec_conf),
#ifndef ALWAYS_USE_MANAGED_MEMORY
          m_fallback(from.m_fallback),
#endif
          m_num_elements(from.m_num_elements), m_pitch(from.m_pitch), m_height(from.m_height),
          m_acquired(false), m_tag(from.m_tag), m_align_bytes(from.m_align_bytes),
          m_is_managed(false)
        {
        if (from.m_data.get())
            {
            allocate();

#ifdef ENABLE_HIP
            if (this->m_exec_conf && this->m_exec_conf->isCUDAEnabled())
                {
                // synchronize all active GPUs
                auto gpu_map = this->m_exec_conf->getGPUIds();
                for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                    {
                    hipSetDevice(gpu_map[idev]);
                    hipDeviceSynchronize();
                    }
                }
#endif

            std::copy(from.m_data.get(), from.m_data.get() + from.m_num_elements, m_data.get());
            }
        }

    //! = operator
    GlobalArray& operator=(const GlobalArray& rhs) noexcept
        {
        m_exec_conf = rhs.m_exec_conf;
#ifndef ALWAYS_USE_MANAGED_MEMORY
        m_fallback = rhs.m_fallback;
#endif

        if (&rhs != this)
            {
            m_num_elements = rhs.m_num_elements;
            m_pitch = rhs.m_pitch;
            m_height = rhs.m_height;
            m_acquired = false;
            m_align_bytes = rhs.m_align_bytes;
            m_tag = rhs.m_tag;

            if (rhs.m_data.get())
                {
                allocate();

#ifdef ENABLE_HIP
                if (this->m_exec_conf && this->m_exec_conf->isCUDAEnabled())
                    {
                    // synchronize all active GPUs
                    auto gpu_map = this->m_exec_conf->getGPUIds();
                    for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                        {
                        hipSetDevice(gpu_map[idev]);
                        hipDeviceSynchronize();
                        }
                    }
#endif

                std::copy(rhs.m_data.get(), rhs.m_data.get() + rhs.m_num_elements, m_data.get());
                }
            else
                {
                m_data.reset();
                }
            }

        return *this;
        }

    //! Move constructor, provided for convenience, so std::swap can be used
    GlobalArray(GlobalArray&& other) noexcept
        : m_exec_conf(std::move(other.m_exec_conf)),
#ifndef ALWAYS_USE_MANAGED_MEMORY
          m_fallback(std::move(other.m_fallback)),
#endif
          m_data(std::move(other.m_data)), m_num_elements(std::move(other.m_num_elements)),
          m_pitch(std::move(other.m_pitch)), m_height(std::move(other.m_height)),
          m_acquired(std::move(other.m_acquired)), m_tag(std::move(other.m_tag)),
          m_align_bytes(std::move(other.m_align_bytes)), m_is_managed(std::move(other.m_is_managed))
#ifdef ENABLE_HIP
          ,
          m_event(std::move(other.m_event))
#endif
        {
        }

    //! Move assignment operator
    GlobalArray& operator=(GlobalArray&& other) noexcept
        {
        // call base clas method
        if (&other != this)
            {
            m_exec_conf = std::move(other.m_exec_conf);
#ifndef ALWAYS_USE_MANAGED_MEMORY
            m_fallback = std::move(other.m_fallback);
#endif
            m_data = std::move(other.m_data);
            m_num_elements = std::move(other.m_num_elements);
            m_pitch = std::move(other.m_pitch);
            m_height = std::move(other.m_height);
            m_acquired = std::move(other.m_acquired);
            m_tag = std::move(other.m_tag);
            m_align_bytes = std::move(other.m_align_bytes);
            m_is_managed = std::move(other.m_is_managed);
#ifdef ENABLE_HIP
            m_event = std::move(other.m_event);
#endif
            }

        return *this;
        }

    /*! Allocate a 2D array in managed memory
        \param width Width of the 2-D array to allocate (in elements)
        \param height Number of rows to allocate in the 2D array
        \param exec_conf Shared pointer to the execution configuration for managing CUDA
       initialization and shutdown
     */
    GlobalArray(size_t width,
                size_t height,
                std::shared_ptr<const ExecutionConfiguration> exec_conf,
                bool force_managed = false)
        : m_exec_conf(exec_conf),
#ifndef ALWAYS_USE_MANAGED_MEMORY
          // explicit copy should be elided
          m_fallback((exec_conf->allConcurrentManagedAccess()
                      || (force_managed && exec_conf->isCUDAEnabled()))
                         ? GPUArray<T>()
                         : GPUArray<T>(width, height, exec_conf)),
#endif
          m_height(height), m_acquired(false), m_align_bytes(0),
          m_is_managed(exec_conf->allConcurrentManagedAccess()
                       || (force_managed && exec_conf->isCUDAEnabled()))
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!m_is_managed)
            return;
#endif

        // make m_pitch the next multiple of 16 larger or equal to the given width
        m_pitch = (width + (16 - (width & 15)));

        m_num_elements = m_pitch * m_height;

#ifdef ENABLE_HIP
        if (this->m_exec_conf->isCUDAEnabled())
            {
            // use OS page size as minimum alignment
            m_align_bytes = getpagesize();
            }
#endif

        if (m_num_elements > 0)
            allocate();
        }

    //! Swap the pointers of two GlobalArrays
    inline void swap(GlobalArray& from)
        {
        if (from.m_acquired || m_acquired)
            {
            throw std::runtime_error("Cannot swap arrays in use.");
            }

        std::swap(m_exec_conf, from.m_exec_conf);
        std::swap(m_num_elements, from.m_num_elements);
        std::swap(m_data, from.m_data);
        std::swap(m_pitch, from.m_pitch);
        std::swap(m_height, from.m_height);
        std::swap(m_tag, from.m_tag);
        std::swap(m_align_bytes, from.m_align_bytes);
        std::swap(m_is_managed, from.m_is_managed);
#ifdef ENABLE_HIP
        std::swap(m_event, from.m_event);
#endif

#ifndef ALWAYS_USE_MANAGED_MEMORY
        m_fallback.swap(from.m_fallback);
#endif
        }

    //! Get the underlying raw pointer
    /*! \returns the content of the underlying smart pointer

        \warning This method doesn't sync the device, so if you are using the pointer to read from
       while a kernel is writing to it on some stream, this may cause undefined behavior

        It may be used to pass the pointer to API functions, e.g., to set memory hints or prefetch
       data asynchronously
     */
    const T* get() const
        {
        return m_data.get();
        }

    //! Get the number of elements
    /*!
     - For 1-D allocated GPUArrays, this is the number of elements allocated.
     - For 2-D allocated GPUArrays, this is the \b total number of elements (\a pitch * \a height)
     allocated
    */
    inline size_t getNumElements() const
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!this->m_exec_conf || !m_is_managed)
            return m_fallback.getNumElements();
#endif

        return m_num_elements;
        }

    //! Test if the GPUArray is NULL
    inline bool isNull() const
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!this->m_exec_conf || !m_is_managed)
            return m_fallback.isNull();
#endif

        return !m_data;
        }

    //! Get the width of the allocated rows in elements
    /*!
     - For 2-D allocated GPUArrays, this is the total width of a row in memory (including the
     padding added for coalescing)
     - For 1-D allocated GPUArrays, this is the simply the number of elements allocated.
    */
    inline size_t getPitch() const
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!this->m_exec_conf || !m_is_managed)
            return m_fallback.getPitch();
#endif

        return m_pitch;
        }

    //! Get the number of rows allocated
    /*!
     - For 2-D allocated GPUArrays, this is the height given to the constructor
     - For 1-D allocated GPUArrays, this is the simply 1.
    */
    inline size_t getHeight() const
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!this->m_exec_conf || !m_is_managed)
            return m_fallback.getHeight();
#endif

        return m_height;
        }

    //! Resize the GlobalArray
    /*! This method resizes the array by allocating a new array and copying over the elements
        from the old array. Resizing is a slow operation.
    */
    inline void resize(size_t num_elements)
        {
#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!this->m_exec_conf || !m_is_managed)
            {
            m_fallback.resize(num_elements);
            this->outputRepresentation();
            return;
            }
#endif

        if (m_acquired)
            {
            throw std::runtime_error("Cannot resize array in use.");
            }

#ifdef ENABLE_HIP
        if (this->m_exec_conf && this->m_exec_conf->isCUDAEnabled())
            {
            // synchronize all active GPUs
            auto gpu_map = this->m_exec_conf->getGPUIds();
            for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                hipSetDevice(gpu_map[idev]);
                hipDeviceSynchronize();
                }
            }
#endif

        // store old data in temporary vector
        std::vector<T> old(m_num_elements);
        std::copy(m_data.get(), m_data.get() + m_num_elements, old.begin());

        size_t num_copy_elements = m_num_elements > num_elements ? num_elements : m_num_elements;

        m_num_elements = num_elements;

        assert(m_num_elements > 0);

        allocate();

        std::copy(old.begin(), old.begin() + num_copy_elements, m_data.get());

        m_pitch = m_num_elements;
        m_height = 1;

        outputRepresentation();
        }

    //! Resize a 2D GlobalArray
    inline void resize(size_t width, size_t height)
        {
        assert(this->m_exec_conf);

#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!m_is_managed)
            {
            m_fallback.resize(width, height);
            outputRepresentation();
            return;
            }
#endif

        if (m_acquired)
            {
            throw std::runtime_error("Cannot resize array in use.");
            }

        // make m_pitch the next multiple of 16 larger or equal to the given width
        size_t pitch = (width + (16 - (width & 15)));

#ifdef ENABLE_HIP
        if (this->m_exec_conf->isCUDAEnabled())
            {
            // synchronize all active GPUs
            auto gpu_map = this->m_exec_conf->getGPUIds();
            for (int idev = this->m_exec_conf->getNumActiveGPUs() - 1; idev >= 0; --idev)
                {
                hipSetDevice(gpu_map[idev]);
                hipDeviceSynchronize();
                }
            }
#endif

        // store old data in temporary vector
        std::vector<T> old(m_num_elements);
        std::copy(m_data.get(), m_data.get() + m_num_elements, old.begin());

        m_num_elements = pitch * height;

        assert(m_num_elements > 0);

        allocate();

        // copy over data
        // every column is copied separately such as to align with the new pitch
        size_t num_copy_rows = m_height > height ? height : m_height;
        size_t num_copy_columns = m_pitch > pitch ? pitch : m_pitch;
        for (size_t i = 0; i < num_copy_rows; i++)
            std::copy(old.begin() + i * m_pitch,
                      old.begin() + i * m_pitch + num_copy_columns,
                      m_data.get() + i * pitch);

        m_height = height;
        m_pitch = pitch;
        outputRepresentation();
        }

    //! Set an optional tag for memory profiling
    /*! tag The name of this allocation
     */
    inline void setTag(const std::string& tag)
        {
        // update the tag
        m_tag = tag;

        // set tag on deleter so it can be displayed upon free
        if (!isNull() && m_data)
            m_data.get_deleter().setTag(tag);

        // for debugging
        this->outputRepresentation();
        }

    //! Return a string representation of this array
    inline std::string getRepresentation() const
        {
        std::ostringstream o;
        if (m_tag != "")
            {
            o << m_tag << ": ";
            }
        else
            {
            o << "anonymous: ";
            }

#ifndef ALWAYS_USE_MANAGED_MEMORY
        if (!m_is_managed)
            {
            o << m_fallback.getRepresentation();
            }
        else
#endif
            {
            if (!isNull())
                {
                const std::string type_name = typeid(T).name();
                int status;
                char* realname = abi::__cxa_demangle(type_name.c_str(), 0, 0, &status);
                if (status)
                    throw std::runtime_error("Status " + std::to_string(status)
                                             + " while trying to demangle data type.");

                auto range = m_data.get_deleter().getAllocationRange();
                o << range.first << "-" << range.second;
                o << " [" << realname << "]";
                free(realname);
                }
            else
                {
                o << "null";
                }
            }
        return o.str();
        }

    inline void outputRepresentation()
        {
        if (m_exec_conf)
            m_exec_conf->msg->notice(9) << getRepresentation() << std::endl;
        }

    protected:
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

    //! Need to be friends with ArrayHandle
    friend class ArrayHandle<T>;
    friend class ArrayHandleAsync<T>;

    friend class GPUArrayBase<T, GlobalArray<T>>;

    friend class GlobalArrayDispatch<T>;

    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< execution configuration for working with CUDA

    private:
#ifndef ALWAYS_USE_MANAGED_MEMORY
    //! We hold a GPUArray<T> object for fallback onto zero-copy memory
    GPUArray<T> m_fallback;
#endif

    std::unique_ptr<T, hoomd::detail::managed_deleter<T>>
        m_data; //!< Smart ptr to managed or host memory, with custom deleter

    size_t m_num_elements; //!< Number of elements in array
    size_t m_pitch;        //!< Pitch of 2D array
    size_t m_height;       //!< Height of 2D array

    mutable bool m_acquired; //!< Tracks if the array is already acquired

    std::string m_tag; //!< Name tag of this buffer (optional)

    size_t m_align_bytes; //!< Size of alignment in bytes
    bool m_is_managed;    //!< Whether or not this array is stored using managed memory.

#ifdef ENABLE_HIP
    std::unique_ptr<hipEvent_t, hoomd::detail::event_deleter>
        m_event; //! CUDA event for synchronization
#endif

    //! Allocate the managed array and construct the items
    void allocate()
        {
        assert(m_num_elements);

        void* ptr = nullptr;
        void* allocation_ptr = nullptr;
        bool use_device = this->m_exec_conf && this->m_exec_conf->isCUDAEnabled();
        size_t allocation_bytes;

#ifdef ENABLE_HIP
        if (use_device)
            {
            // Check for pending errors.
            CHECK_CUDA_ERROR();

            allocation_bytes = m_num_elements * sizeof(T);

            // always reserve an extra page
            if (m_align_bytes)
                allocation_bytes += m_align_bytes;

            this->m_exec_conf->msg->notice(10)
                << "Allocating " << allocation_bytes << " bytes of managed memory." << std::endl;

            hipError_t error = hipMallocManaged(&ptr, allocation_bytes, hipMemAttachGlobal);
            if (error == hipErrorMemoryAllocation)
                {
                throw std::bad_alloc();
                }
            else if (error != hipSuccess)
                {
                throw std::runtime_error(hipGetErrorString(error));
                }

            allocation_ptr = ptr;

            if (m_align_bytes)
                {
// align to align_size
#ifndef NO_STD_ALIGN
                ptr = std::align(m_align_bytes, m_num_elements * sizeof(T), ptr, allocation_bytes);
#else
                ptr = hoomd::detail::my_align(m_align_bytes,
                                              m_num_elements * sizeof(T),
                                              ptr,
                                              allocation_bytes);
#endif

                if (!ptr)
                    throw std::bad_alloc();
                }
            }
        else
#endif
            {
            int retval = posix_memalign((void**)&ptr, 32, m_num_elements * sizeof(T));
            if (retval != 0)
                {
                throw std::bad_alloc();
                }
            allocation_bytes = m_num_elements * sizeof(T);
            allocation_ptr = ptr;
            }

#ifdef ENABLE_HIP
        if (use_device)
            {
            hipDeviceSynchronize();
            CHECK_CUDA_ERROR();
            }

        if (use_device)
            {
            m_event = std::unique_ptr<hipEvent_t, hoomd::detail::event_deleter>(
                new hipEvent_t,
                hoomd::detail::event_deleter(this->m_exec_conf));

            hipEventCreateWithFlags(m_event.get(), hipEventDisableTiming);
            CHECK_CUDA_ERROR();
            }
#endif

        // store allocation and custom deleter in unique_ptr
        hoomd::detail::managed_deleter<T> deleter(this->m_exec_conf,
                                                  use_device,
                                                  m_num_elements,
                                                  allocation_ptr,
                                                  allocation_bytes);
        deleter.setTag(m_tag);
        m_data = std::unique_ptr<T, decltype(deleter)>(reinterpret_cast<T*>(ptr), deleter);

        // construct objects explicitly using placement new
        for (std::size_t i = 0; i < m_num_elements; ++i)
            ::new ((void**)&((T*)ptr)[i]) T;

        // display representation for debugging
        if (m_tag != "")
            outputRepresentation();
        }
    };

//************************************************
// ArrayHandleDispatch specialization for GPUArray
// ***********************************************

template<class T> class GlobalArrayDispatch : public ArrayHandleDispatch<T>
    {
    public:
    GlobalArrayDispatch(T* const _data, const GlobalArray<T>& _global_array)
        : ArrayHandleDispatch<T>(_data), global_array(_global_array)
        {
        }

    virtual ~GlobalArrayDispatch()
        {
        assert(global_array.isAcquired());
        global_array.release();
        }

    private:
    const GlobalArray<T>& global_array;
    };

// ***********************************************
// GlobalArray implementation
// ***********************************************

template<class T>
inline ArrayHandleDispatch<T> GlobalArray<T>::acquire(const access_location::Enum location,
                                                      const access_mode::Enum mode
#ifdef ENABLE_HIP
                                                      ,
                                                      bool async
#endif
) const

    {
#ifndef ALWAYS_USE_MANAGED_MEMORY
    if (!this->m_exec_conf || !m_is_managed)
        return m_fallback.acquire(location,
                                  mode
#ifdef ENABLE_HIP
                                  ,
                                  async
#endif
        );
#endif

    if (m_acquired)
        {
        throw std::runtime_error("Cannot acquire access to array in use [" + this->m_tag + "]");
        }
    m_acquired = true;

    // make sure a null array can be acquired
    if (!this->m_exec_conf || isNull())
        return GlobalArrayDispatch<T>(nullptr, *this);

    if (this->m_exec_conf && this->m_exec_conf->inMultiGPUBlock())
        {
        // we throw this error because we are not syncing all GPUs upon acquire
        throw std::runtime_error("GlobalArray should not be acquired in a multi-GPU block.");
        }

#ifdef ENABLE_HIP
    bool use_device = this->m_exec_conf && this->m_exec_conf->isCUDAEnabled();
    if (!isNull() && use_device && location == access_location::host && !async)
        {
        // synchronize GPU 0
        hipEventRecord(*m_event);
        hipEventSynchronize(*m_event);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
#endif

    return GlobalArrayDispatch<T>(m_data.get(), *this);
    }
    } // end namespace hoomd
