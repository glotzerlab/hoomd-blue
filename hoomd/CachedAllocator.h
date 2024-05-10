// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CachedAllocator.h
    \brief Declares a cached allocator for temporary allocations and a helper class

    Inspired by thrust/examples/cuda/custom_temporary_allocation.cu
*/

#ifndef __CACHED_ALLOCATOR_H__
#define __CACHED_ALLOCATOR_H__

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>

#include <cassert>
#include <map>
#include <stdexcept>

//! Need to define an error checking macro that can be used in .cu files
#define CHECK_CUDA()                                                         \
        {                                                                    \
        hipError_t err = hipDeviceSynchronize();                             \
        if (err != hipSuccess)                                               \
            {                                                                \
            throw std::runtime_error("CUDA Error in CachedAllocator "        \
                                     + std::string(hipGetErrorString(err))); \
            }                                                                \
        err = hipGetLastError();                                             \
        if (err != hipSuccess)                                               \
            {                                                                \
            throw std::runtime_error("CUDA Error in CachedAllocator "        \
                                     + std::string(hipGetErrorString(err))); \
            }                                                                \
        }

namespace hoomd
    {
//! CachedAllocator: a simple allocator for caching allocation requests
class __attribute__((visibility("default"))) CachedAllocator
    {
    public:
    // needed by thrust
    typedef char value_type;

    //! Constructor
    /*  \param max_cached_bytes Maximum size of cache
     *   \param cache_reltol Relative tolerance for cache hits
     */
    CachedAllocator(bool managed,
                    unsigned int max_cached_bytes = 100u * 1024u * 1024u,
                    float cache_reltol = 0.1f)
        : m_managed(managed), m_num_bytes_tot(0), m_max_cached_bytes(max_cached_bytes),
          m_cache_reltol(cache_reltol)
        {
        }

    CachedAllocator(const CachedAllocator&) = delete;
    CachedAllocator& operator=(const CachedAllocator&) = delete;

    CachedAllocator(const CachedAllocator&&) = delete;
    CachedAllocator& operator=(const CachedAllocator&&) = delete;

    //! Set maximum cache size
    void setMaxCachedBytes(unsigned int max_cached_bytes)
        {
        m_max_cached_bytes = max_cached_bytes;
        }

    //! Destructor
    virtual ~CachedAllocator()
        {
        // free all allocations when cached_allocator goes out of scope
        free_all();
        }

    //! Allocate a temporary block
    /*! \param num_bytes Number of elements to allocate
     * \returns a pointer to the allocated buffer
     */
    template<typename T> T* getTemporaryBuffer(size_t num_elements);

    // Specifically allocate a char* buffer
    char* allocate(std::ptrdiff_t num_bytes)
        {
        return getTemporaryBuffer<char>(num_bytes);
        }

    //! Release a previously allocated block
    void deallocate(char* ptr, size_t n = 0)
        {
        if (ptr == NULL)
            return;

        // erase the allocated block from the allocated blocks map
        allocated_blocks_type::iterator iter = m_allocated_blocks.find(ptr);
        assert(iter != m_allocated_blocks.end());
        std::ptrdiff_t num_bytes = iter->second;
        m_allocated_blocks.erase(iter);

        // insert the block into the free blocks map
        m_free_blocks.insert(std::make_pair(num_bytes, ptr));
        }

    private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

    bool m_managed; //! True if we use unified memory

    size_t m_num_bytes_tot;
    size_t m_max_cached_bytes;
    float m_cache_reltol;

    free_blocks_type m_free_blocks;
    allocated_blocks_type m_allocated_blocks;

    //! Free all allocated blocks
    void free_all()
        {
        //            m_exec_conf->msg->notice(5) << "CachedAllocator: Cleaning up after ourselves"
        //                << std::endl;

        // deallocate all outstanding blocks in both lists
        for (free_blocks_type::iterator i = m_free_blocks.begin(); i != m_free_blocks.end(); ++i)
            {
            hipFree((void*)i->second);
            }

        for (allocated_blocks_type::iterator i = m_allocated_blocks.begin();
             i != m_allocated_blocks.end();
             ++i)
            {
            hipFree((void*)i->first);
            }
        }
    };

//! A temporary allocation object
/*! When it goes out of scope, the temporary memory is automatically returned to the cache
 */
template<typename T> class ScopedAllocation
    {
    public:
    //! Copy constructor
    ScopedAllocation(CachedAllocator& alloc, size_t num_elements);

    //! Destructor
    ~ScopedAllocation();

    T* operator()()
        {
        return data;
        }

    T* data;

    private:
    ScopedAllocation(char* ptr, CachedAllocator& alloc);

    CachedAllocator& m_alloc;

    friend class CachedAllocator;
    };

template<typename T> T* CachedAllocator::getTemporaryBuffer(size_t num_elements)
    {
    std::ptrdiff_t num_bytes = sizeof(T) * num_elements;
    char* result = 0;

    // short-cut to avoid storing duplicate NULL ptrs in the map
    if (!num_bytes)
        return (T*)NULL;

    size_t num_allocated_bytes = num_bytes;

    // search the cache for a free block
    free_blocks_type::iterator free_block = m_free_blocks.lower_bound(num_bytes);

    // look for a cached buffer within m_cache_reltol relative tolerance
    if (free_block != m_free_blocks.end()
        && free_block->first <= (num_bytes + (unsigned int)((float)num_bytes * m_cache_reltol)))
        {
        //        m_exec_conf->msg->notice(10) << "CachedAllocator: found a hit "
        //            << "(" << float(num_bytes)/1024.0f/1024.0f << " MB)" << std::endl;

        // get the pointer
        result = free_block->second;

        num_allocated_bytes = free_block->first;

        // erase from the free_blocks map
        m_free_blocks.erase(free_block);
        }
    else
        {
        // no allocation of the right size exists
        // create a new one with cudaMalloc
        //        m_exec_conf->msg->notice(10) << "CachedAllocator: no free block found;"
        //            << " allocating " << float(num_bytes)/1024.0f/1024.0f << " MB" << std::endl;

        if (m_managed)
            hipMallocManaged((void**)&result, num_bytes);
        else
            hipMalloc((void**)&result, num_bytes);
        CHECK_CUDA();

        m_num_bytes_tot += num_bytes;

        while (m_num_bytes_tot > m_max_cached_bytes && m_free_blocks.size())
            {
            // eliminate largest cached block
            free_blocks_type::reverse_iterator i = m_free_blocks.rbegin();

            //            m_exec_conf->msg->notice(10) << "CachedAllocator: maximum cache size "
            //                << "reached; removing unused block ("
            //                << float(i->first)/1024.0f/1024.0f << " MB)" << std::endl;

            hipFree((void*)i->second);

            CHECK_CUDA();
            m_num_bytes_tot -= i->first;

            m_free_blocks.erase((++i).base());
            }
        }

    // insert the allocated pointer into the allocated_blocks map
    m_allocated_blocks.insert(std::make_pair(result, num_allocated_bytes));

    return (T*)result;
    }

//! Constructor
template<typename T>
ScopedAllocation<T>::ScopedAllocation(CachedAllocator& alloc, size_t num_elements) : m_alloc(alloc)
    {
    data = m_alloc.getTemporaryBuffer<T>(num_elements);
    }

//! Destructor
template<typename T> ScopedAllocation<T>::~ScopedAllocation()
    {
    m_alloc.deallocate((char*)data);
    }

    } // end namespace hoomd

#undef CHECK_CUDA
#endif // ENABLE_HIP
#endif // __CACHED_ALLOCATOR_H__
