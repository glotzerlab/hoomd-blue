// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

//! Class to perform cudaMallocManaged allocations
//! Adapted from github.com/jaredhoberock/managed_allocator

#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>

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

            #ifdef ENABLE_CUDA
            if (m_use_device)
                {
                cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }
                }
            else
            #endif
                {

                int retval = posix_memalign((void **) &result, 32, n*sizeof(T));
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating aligned memory");
                    }
                }

            return result;
            }

        // Static version, also constructs objects
        static value_type *allocate_construct(std::size_t n, bool use_device)
            {
            value_type *result = nullptr;

            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaMallocManaged(&result, (int) (n*sizeof(T)), cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }
                }
            else
            #endif
                {
                int retval = posix_memalign((void **) &result, 32, n*sizeof(T));
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating aligned memory");
                    }
                }

            // construct objects explicitly using placement new
            for (std::size_t i = 0; i < n; ++i) ::new ((void **) &result[i]) value_type;

            return result;
            }


        void deallocate(value_type *ptr, std::size_t N)
            {
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

        //! Static version, also destroys objects
        static void deallocate_destroy(value_type *ptr, std::size_t N, bool use_device)
            {
            // we used placement new in the allocation, so call destructors explicitly
            for (std::size_t i = 0; i < N; ++i)
                {
                ptr[i].~value_type();
                }

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
                free(ptr);
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
