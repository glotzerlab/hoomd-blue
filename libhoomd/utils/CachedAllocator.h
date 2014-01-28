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

/*! \file CachedAllocator.h
    \brief Declares a cached allocator for temporary allocations and a helper class

    Inspired by thrust/examples/cuda/custom_temporary_allocation.cu
*/

#ifndef __CACHED_ALLOCATOR_H__
#define __CACHED_ALLOCATOR_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_CUDA
#include "ExecutionConfiguration.h"

#include <cuda_runtime.h>

#include <map>
#include <cassert>

//! CachedAllocator: a simple allocator for caching allocation requests
class CachedAllocator
    {
    public:
        //! Constructor
        /*! \param exec_conf HOOMD execution configuration
         *  \param max_cached_bytes Maximum size of cache
         *  \param cache_reltol Relative tolerance for cache hits
         */
        CachedAllocator(const ExecutionConfiguration *exec_conf,
            unsigned int max_cached_bytes=100*1024*1024, float cache_reltol = 0.1f)
            : m_exec_conf(exec_conf),
              m_num_bytes_tot(0),
              m_max_cached_bytes(max_cached_bytes),
              m_cache_reltol(cache_reltol)
            { }

        //! Set maximum cache size
        void setMaxCachedBytes(unsigned int max_cached_bytes)
            {
            m_max_cached_bytes = max_cached_bytes;
            }

        //! Destructor
        ~CachedAllocator()
        {
          // free all allocations when cached_allocator goes out of scope
          free_all();
        }

        //! Allocate a temporary block
        /*! \param num_bytes Number of elements to allocate
         * \returns a pointer to the allocated buffer
         */
        template<typename T>
        T *getTemporaryBuffer(unsigned int num_elements) const;

        //! Release a previously allocated block
        void deallocate(char *ptr) const
            {
            if (ptr == NULL) return;

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
        typedef std::map<char *, std::ptrdiff_t> allocated_blocks_type;

        const ExecutionConfiguration *m_exec_conf;
        mutable unsigned int m_num_bytes_tot;
        unsigned int m_max_cached_bytes;
        float m_cache_reltol;

        mutable free_blocks_type m_free_blocks;
        mutable allocated_blocks_type m_allocated_blocks;

        //! Free all allocated blocks
        void free_all() const
            {
            m_exec_conf->msg->notice(5) << "CachedAllocator: Cleaning up after ourselves"
                << std::endl;

            // deallocate all outstanding blocks in both lists
            for(free_blocks_type::iterator i = m_free_blocks.begin(); i != m_free_blocks.end(); ++i)
                {
                cudaFree((void *) i->second);
                CHECK_CUDA_ERROR();
                }

            for(allocated_blocks_type::iterator i = m_allocated_blocks.begin();
                i != m_allocated_blocks.end(); ++i)
                {
                cudaFree((void *) i->first);
                CHECK_CUDA_ERROR();
                }
            }
    };

//! A temporary allocation object
/*! When it goes out of scope, the temporary memory is automatically returned to the cache
 */
template<typename T>
class ScopedAllocation
    {
    public:
        //! Copy constructor
        ScopedAllocation(const CachedAllocator& alloc, unsigned int num_elements);

        //! Destructor
        ~ScopedAllocation();

        T *operator() ()
            {
            return data;
            }

        T * data;

    private:
        ScopedAllocation(char *ptr, CachedAllocator& alloc);

        const CachedAllocator& m_alloc;

        friend class CachedAllocator;
    };

template<typename T>
T* CachedAllocator::getTemporaryBuffer(unsigned int num_elements) const
    {
    std::ptrdiff_t num_bytes = sizeof(T)*num_elements;
    char *result = 0;

    // short-cut to avoid storing duplicate NULL ptrs in the map
    if (!num_bytes) return (T*)NULL;

    unsigned int num_allocated_bytes = num_bytes;

    // search the cache for a free block
    free_blocks_type::iterator free_block = m_free_blocks.lower_bound(num_bytes);

    // look for a cached buffer within m_cache_reltol relative tolerance
    if(free_block != m_free_blocks.end()
         && free_block->first <= (num_bytes + (unsigned int)((float)num_bytes*m_cache_reltol)))
        {
        m_exec_conf->msg->notice(10) << "CachedAllocator: found a hit "
            << "(" << float(num_bytes)/1024.0f/1024.0f << " MB)" << std::endl;

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
        m_exec_conf->msg->notice(10) << "CachedAllocator: no free block found;"
            << " allocating " << float(num_bytes)/1024.0f/1024.0f << " MB" << std::endl;

        cudaMalloc((void **) &result, num_bytes);
        CHECK_CUDA_ERROR();

        m_num_bytes_tot += num_bytes;

        while (m_num_bytes_tot > m_max_cached_bytes && m_free_blocks.size())
            {
            // eliminate largest cached block
            free_blocks_type::reverse_iterator i = m_free_blocks.rbegin();

            m_exec_conf->msg->notice(10) << "CachedAllocator: maximum cache size "
                << "reached; removing unused block ("
                << float(i->first)/1024.0f/1024.0f << " MB)" << std::endl;

            // transform the pointer to cuda::pointer before calling cuda::free
            cudaFree((void *) i->second);
            CHECK_CUDA_ERROR();
            m_num_bytes_tot -= i->first;

            m_free_blocks.erase((++i).base());
            }
        }

    // insert the allocated pointer into the allocated_blocks map
    m_allocated_blocks.insert(std::make_pair(result, num_allocated_bytes));

    return (T *)result;
    }

//! Constructor
template<typename T>
ScopedAllocation<T>::ScopedAllocation(const CachedAllocator& alloc, unsigned int num_elements)
    : m_alloc(alloc)
    {
    data = m_alloc.getTemporaryBuffer<T>(num_elements);
    }

//! Destructor
template<typename T>
ScopedAllocation<T>::~ScopedAllocation()
    {
    m_alloc.deallocate((char *) data);
    }


#endif // ENABLE_CUDA
#endif // __CACHED_ALLOCATOR_H__
