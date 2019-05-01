// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

//! Class to perform cudaMallocManaged allocations
//! Adapted from github.com/jaredhoberock/managed_allocator

#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <memory>

#ifdef __GNUC__
#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)
/* Test for GCC < 5.0 */
#if GCC_VERSION < 50000
// work around GCC missing feature

#define NO_STD_ALIGN
// https://stackoverflow.com/questions/27064791/stdalign-not-supported-by-g4-9
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57350
inline void *my_align( std::size_t alignment, std::size_t size,
                    void *&ptr, std::size_t &space ) {
    std::uintptr_t pn = reinterpret_cast< std::uintptr_t >( ptr );
    std::uintptr_t aligned = ( pn + alignment - 1 ) & - alignment;
    std::size_t padding = aligned - pn;
    if ( space < size + padding ) return nullptr;
    space -= padding;
    return ptr = reinterpret_cast< void * >( aligned );
    }
#endif
#endif

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
            void *result = nullptr;

            #ifdef ENABLE_CUDA
            if (m_use_device)
                {
                size_t allocation_bytes = n*sizeof(T);

                cudaError_t error = cudaMallocManaged(&result, allocation_bytes, cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }
                }
            else
            #endif
                {

                int retval = posix_memalign(&result, 32, n*sizeof(T));
                if (retval != 0)
                    {
                    throw std::runtime_error("Error allocating aligned memory");
                    }
                }

            return (value_type *) result;
            }

        // Static version, also constructs objects
        /*! \param n Number of elements to allote
            \param use_device Whether to use cudaMallocManaged
            \param align_size Number of bytes to align start of allocation to
            \param allocation_bytes (Return value) Size of total allocation (including aligned part)
            \param allocation_ptr (Return value) Start of un-aligned allocation
         */
        static value_type *allocate_construct_aligned(std::size_t n, bool use_device, size_t align_size,
            size_t &allocation_bytes, void *&allocation_ptr)
            {
            void *result = nullptr;

            #ifdef ENABLE_CUDA
            if (use_device)
                {
                allocation_bytes = n*sizeof(T);

                if (align_size)
                    allocation_bytes = ((n*sizeof(T))/align_size + 1)*align_size;

                cudaError_t error = cudaMallocManaged(&result, allocation_bytes, cudaMemAttachGlobal);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error allocating managed memory");
                    }

                allocation_ptr = result;

                if (align_size)
                    {
                    // align to align_size
                    #ifndef NO_STD_ALIGN
                    result = std::align(align_size,n*sizeof(T),result,allocation_bytes);
                    #else
                    result = my_align(align_size,n*sizeof(T),result,allocation_bytes);
                    #endif

                    if (!result)
                        throw std::runtime_error("managed_allocator: Error aligning managed memory");
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
                allocation_bytes = n*sizeof(T);
                allocation_ptr = result;
                }

            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaDeviceSynchronize();
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error on device sync during allocate_construct");
                    }
                }
            #endif

            // construct objects explicitly using placement new
            for (std::size_t i = 0; i < n; ++i) ::new ((void **) &((value_type *)result)[i]) value_type;

            return (value_type *)result;
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
        /*! \param ptr Start of aligned memory allocation
            \param N Number elements allocated
            \param use_device Whether this is a CUDA managed memory allocation
            \param allocation_ptr address of unaligned allocation that includes the aligned portion
         */
        static void deallocate_destroy_aligned(value_type *ptr, std::size_t N, bool use_device,
            void *allocation_ptr)
            {
            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaDeviceSynchronize();
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error on device sync during deallocate_destroy");
                    }
                }
            #endif

            // we used placement new in the allocation, so call destructors explicitly
            for (std::size_t i = 0; i < N; ++i)
                {
                ptr[i].~value_type();
                }

            #ifdef ENABLE_CUDA
            if (use_device)
                {
                cudaError_t error = cudaFree(allocation_ptr);
                if (error != cudaSuccess)
                    {
                    std::cerr << cudaGetErrorString(error) << std::endl;
                    throw std::runtime_error("managed_allocator: Error freeing managed memory");
                    }
                }
            else
            #endif
                {
                free(allocation_ptr);
                }
            }


        bool usesDevice() const { return m_use_device; };

    private:
        bool m_use_device;   //!< Whether to use cudaMallocManaged
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
