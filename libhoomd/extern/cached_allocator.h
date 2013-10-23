// Maintainer: jglaser

/*! \file CachedAllocator.cuh
    \brief Implements a cached allocator for thrust code

    See thrust/examples/cuda/custom_temporary_allocation.cu
*/

#ifndef __CACHED_ALLOCATOR_H__
#define __CACHED_ALLOCATOR_H__

#ifdef ENABLE_CUDA
#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <map>
#include <assert.h>

// 100 MB max cache
#define MAX_CACHED_BYTES 100*1024*1024

// to use a cached allocation, it may be larger than the requested size by a relative tolerance
#define CACHE_RELTOL 0.1

/* From the original documentation:

  This example demonstrates how to intercept calls to get_temporary_buffer
  and return_temporary_buffer to control how Thrust allocates temporary storage
  during algorithms such as thrust::sort. The idea will be to create a simple
  cache of allocations to search when temporary storage is requested. If a hit
  is found in the cache, we quickly return the cached allocation instead of
  resorting to the more expensive thrust::cuda::malloc.

  Note: this implementation cached_allocator is not thread-safe. If multiple
  (host) threads use the same cached_allocator then they should gain exclusive
  access to the allocator before accessing its methods.
 */


// cached_allocator: a simple allocator for caching allocation requests
class cached_allocator
{
  public:
    // just allocate bytes
    typedef char value_type;

    cached_allocator()
        : num_bytes_tot(0) {}

    ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
      char *result = 0;

      // short-cut to avoid storing duplicate NULL ptrs in the map
      if (!num_bytes) return NULL;

      unsigned int num_allocated_bytes = num_bytes;
      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.lower_bound(num_bytes);

      if(free_block != free_blocks.end()
         && free_block->first <= ((unsigned int)((float)num_bytes*(1.0+CACHE_RELTOL))))
      {
        //std::cout << "cached_allocator::allocator(): found a hit" << std::endl;

        // get the pointer
        result = free_block->second;

        num_allocated_bytes = free_block->first;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
        {
          //std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc ("
          //          << num_bytes << " bytes)" << std::endl;

          // allocate memory and convert cuda::pointer to raw pointer
          result = thrust::cuda::malloc<char>(num_bytes).get();

          num_bytes_tot += num_bytes;

          while (num_bytes_tot > MAX_CACHED_BYTES && free_blocks.size())
            {
            // eliminate first free blocks
            free_blocks_type::iterator i = free_blocks.begin();

            // transform the pointer to cuda::pointer before calling cuda::free
            thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
            num_bytes_tot -= i->first;

            free_blocks.erase(i);
            }
        }
        catch(std::runtime_error &e)
        {
          throw;
        }
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_allocated_bytes));

      return result;
    }

    void deallocate(char *ptr, size_t n)
    {
      if (ptr == NULL) return;

      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      assert(iter != allocated_blocks.end());
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

  private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char *, std::ptrdiff_t> allocated_blocks_type;

    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;

    unsigned int num_bytes_tot;

    void free_all()
    {
      //std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;

      // deallocate all outstanding blocks in both lists
      for(free_blocks_type::iterator i = free_blocks.begin();
          i != free_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
      }

      for(allocated_blocks_type::iterator i = allocated_blocks.begin();
          i != allocated_blocks.end();
          ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
    }

};

#endif

#endif // __CACHED_ALLOCATOR_H__
