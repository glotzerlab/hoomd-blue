// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

//! Class to perform cudaMallocManaged allocations
//! Adapted from github.com/jaredhoberock/managed_allocator

#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>

// C++ may store internal data at the beginning of the buffer when using new[]
const std::size_t MAX_OVERHEAD_BYTES=32;

// http://stackoverflow.com/questions/8720425/array-placement-new-requires-unspecified-overhead-in-the-buffer
inline void* operator new[](std::size_t n, void* p, std::size_t limit)
    {
    if (n > limit) throw std::bad_alloc();
    return p;
    }

template<class T>
class managed_allocator
    {
    public:
        //! Default constructor
        managed_allocator()
            : m_use_device(false)
            { }

        //! Ctor
        managed_allocator(bool use_device)
            : m_use_device(use_device)
            { }

        using value_type = T;
        using propagate_on_container_copy_assignment = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;

        value_type *allocate(std::size_t n)
            {
            value_type *result = nullptr;

            void *ptr;

            #ifdef ENABLE_CUDA
            if (m_use_device)
                {
                cudaError_t error = cudaMallocManaged(&ptr, n*sizeof(T)+MAX_OVERHEAD_BYTES, cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }
                }
            else
            #endif
                {

                int retval = posix_memalign(&ptr, 32, n*sizeof(T)+MAX_OVERHEAD_BYTES);
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating aligned memory");
                    }
                }
            // construct objects using placement new
            result = ::new(ptr,n*sizeof(T)+MAX_OVERHEAD_BYTES) value_type[n];
            return result;
            }

        // Static version
        static value_type *allocate(std::size_t n, bool use_device)
            {
            value_type *result = nullptr;
            void *ptr;

            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaMallocManaged(&ptr, n*sizeof(T)+MAX_OVERHEAD_BYTES, cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }
                }
            else
            #endif
                {
                int retval = posix_memalign(&ptr, 32, n*sizeof(T)+MAX_OVERHEAD_BYTES);
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating aligned memory");
                    }
                }

            // construct objects using placement new
            result = ::new(ptr,n*sizeof(T)+MAX_OVERHEAD_BYTES) value_type[n];

            return result;
            }


        void deallocate(value_type *ptr, std::size_t N)
            {
            // we used placement new in the allocation, so call destructors explicitly
            for (std::size_t i = 0; i < N; ++i) ptr[i].~value_type();

            #ifdef ENABLE_CUDA
            if (m_use_device)
                {
                cudaError_t error = cudaFree(ptr);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error freeing managed memory");
                    }
                }
            else
            #endif
                {
                free(ptr);
                }
            }

        //! Static version
        static void deallocate(value_type *ptr, std::size_t N, bool use_device)
            {
            // we used placement new in the allocation, so call destructors explicitly
            for (std::size_t i = 0; i < N; ++i)
                {
                printf("Calling destructor %d\n",(int) i);
                ptr[i].~value_type();
                }

            printf("All destructors called.\n");
            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaFree(ptr);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error freeing managed memory");
                    }
                }
            else
            #endif
                {
                printf("Trying to delete %p\n",ptr);
                ::operator delete [](ptr);
                printf("Deleted\n");
                }
            }


        bool usesDevice() const { return m_use_device; };

    private:
        bool m_use_device; //!< Whether to use cudaMallocManaged
    };

template<class T, class U>
bool operator==(const managed_allocator<T>& lhs, const managed_allocator<U>& rhs)
    {
    return lhs.usesDevice() == rhs.usesDevice();
    }

template<class T, class U>
bool operator!=(const managed_allocator<T>& lhs, const managed_allocator<U>& rhs)
    {
    return lhs.usesDevice() != rhs.usesDevice();
    }

