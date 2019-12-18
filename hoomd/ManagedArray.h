// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef __HIPCC__
#include "managed_allocator.h"

#include <algorithm>
#include <utility>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <memory>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

//! A device-side, fixed-size array memory-managed through cudaMallocManaged
template<class T>
class ManagedArray
    {
    public:
        //! Default constructor
        DEVICE ManagedArray()
            : data(nullptr), ptr(nullptr), N(0), managed(0), align(0),
              allocation_ptr(nullptr), allocation_bytes(0)
            { }

        #ifndef __HIPCC__
        ManagedArray(unsigned int _N, bool _managed, size_t _align = 0)
            : data(nullptr), ptr(nullptr), N(_N), managed(_managed), align(_align),
              allocation_ptr(nullptr), allocation_bytes(0)
            {
            if (N > 0)
                {
                allocate();
                }
            }
        #endif

        DEVICE ~ManagedArray()
            {
            #ifndef __HIPCC__
            deallocate();
            #endif
            }

        //! Copy constructor
        /*! \warn the copy constructor reads from the other array and assumes that array is available on the
                  host. If the GPU isn't synced up, this can lead to errors, so proper multi-GPU synchronization
                  needs to be ensured
         */
        DEVICE ManagedArray(const ManagedArray<T>& other)
            : data(nullptr), ptr(nullptr), N(other.N), managed(other.managed), align(other.align),
              allocation_ptr(nullptr), allocation_bytes(0)
            {
            #ifndef __HIPCC__
            if (N > 0)
                {
                allocate();

                std::copy(other.ptr, other.ptr+N, ptr);
                }
            #else
            ptr = other.ptr;
            data = other.data;
            #endif
            }

        //! Assignment operator
        /*! \warn the copy assignment constructor reads from the other array and assumes that array is available on the
                  host. If the GPU isn't synced up, this can lead to errors, so proper multi-GPU synchronization
                  needs to be ensured
         */
        DEVICE ManagedArray& operator=(const ManagedArray<T>& other)
            {
            #ifndef __HIPCC__
            deallocate();
            #endif

            N = other.N;
            managed = other.managed;
            align = other.align;

            #ifndef __HIPCC__
            if (N > 0)
                {
                allocate();

                std::copy(other.ptr, other.ptr+N, ptr);
                }
            #else
            ptr = other.ptr;
            data = other.data;
            #endif

            return *this;
            }


        #ifndef __HIPCC__

        //! Copy data from other array
        /*! Copy data from the other array, but keep our own alignment and managed flags. HPMC uses this when copying
            non-managed python-constructed shape parameters into its own managed data structures.
         */
        DEVICE void copyDataFrom(const ManagedArray<T>& other)
            {
            deallocate();

            N = other.N;
            // keep our values for managed and align

            if (N > 0)
                {
                allocate();

                std::copy(other.ptr, other.ptr+N, ptr);
                }
            }

        #endif

        //! random access operator
        HOSTDEVICE inline T& operator[](unsigned int i)
            {
            return data[i];
            }

        //! random access operator (const version)
        HOSTDEVICE inline const T& operator[](unsigned int i) const
            {
            return data[i];
            }

        //! Get pointer to array data
        HOSTDEVICE inline T * get()
            {
            return data;
            }

        //! Get pointer to array data (const version)
        HOSTDEVICE inline const T* get() const
            {
            return data;
            }

        #ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
            if (managed && ptr)
                {
                #if defined(__HIP_PLATFORM_NVCC__) && (CUDART_VERSION >= 8000)
                cudaMemAdvise(ptr, sizeof(T)*N, cudaMemAdviseSetReadMostly, 0);
                #endif
                }
            }
        #endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        HOSTDEVICE void* allocate_shared(char *& s_ptr, unsigned int &available_bytes) const
            {
            // size in ints (round up)
            unsigned int size_int = (sizeof(T)*N)/sizeof(int);
            if ((sizeof(T)*N) % sizeof(int)) size_int++;

            // align ptr to size of data type
            unsigned long int max_align_bytes = (sizeof(int) > sizeof(T) ? sizeof(int) : sizeof(T))-1;
            char *ptr_align = (char *)(((unsigned long int)s_ptr + max_align_bytes) & ~max_align_bytes);

            if (size_int*sizeof(int)+max_align_bytes > available_bytes)
                return nullptr;

            // increment pointer
            s_ptr = ptr_align + size_int*sizeof(int);
            available_bytes -= size_int*sizeof(int)+max_align_bytes;

            return (void *)ptr_align;
            }

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation

            \returns true if array was loaded into shared memory
         */
        DEVICE bool load_shared(char *& s_ptr, unsigned int &available_bytes)
            {
            // align ptr to size of data type
            void *ptr_align = allocate_shared(s_ptr, available_bytes);

            if (! ptr_align)
                return false;

            #ifdef __HIP_DEVICE_COMPILE__
            // only in GPU code
            unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
            unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

            unsigned int size_int = (sizeof(T)*N)/sizeof(int);
            if ((sizeof(T)*N) % sizeof(int)) size_int++;

            for (unsigned int cur_offset = 0; cur_offset < size_int; cur_offset += block_size)
                {
                if (cur_offset + tidx < size_int)
                    {
                    ((int *)ptr_align)[cur_offset + tidx] = ((int *)ptr)[cur_offset + tidx];
                    }
                }

            // redirect data ptr
            data = (T *) ptr_align;
            #endif

            return true;
            }

        bool isManaged() const
            {
            return managed;
            }

        HOSTDEVICE unsigned int size() const
            {
            return N;
            }

        size_t getAllocationBytes() const
            {
            return allocation_bytes;
            }

    protected:
        #ifndef __HIPCC__
        void allocate()
            {
            ptr = managed_allocator<T>::allocate_construct_aligned(N, managed, align, allocation_bytes, allocation_ptr);
            data = ptr;
            }

        void deallocate()
            {
            if (N > 0)
                {
                managed_allocator<T>::deallocate_destroy_aligned(ptr, N, managed, allocation_ptr);
                }
            }
        #endif

    private:
        T *data;                 //!< Data pointer
        T *ptr;                  //!< Original data pointer
        unsigned int N;          //!< Number of data elements
        unsigned int managed;    //!< True if we are CUDA managed
        size_t align;            //!< Alignment size
        void *allocation_ptr;    //!< Pointer to un-aligned start of allocation
        size_t allocation_bytes; //!< Total size of allocation, including aligned part
    };
