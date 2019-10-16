// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef NVCC
#include "managed_allocator.h"

#include <algorithm>
#include <utility>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>

#ifdef NVCC
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
            : data(nullptr), N(0), managed(0), align(0),
              allocation_ptr(nullptr), allocation_bytes(0)
            { }

        #ifndef NVCC
        ManagedArray(unsigned int _N, bool _managed, size_t _align = 0)
            : data(nullptr), N(_N), managed(_managed), align(_align),
              allocation_ptr(nullptr), allocation_bytes(0)
            {
            if (N > 0)
                {
                allocate();
                }
            }
        #endif

        DEVICE virtual ~ManagedArray()
            {
            #ifndef NVCC
            deallocate();
            #endif
            }

        //! Copy constructor
        /*! \warn the copy constructor reads from the other array and assumes that array is available on the
                  host. If the GPU isn't synced up, this can lead to erros, so proper multi-GPU synchronization
                  needs to be ensured
         */
        DEVICE ManagedArray(const ManagedArray<T>& other)
            : data(nullptr), N(other.N), managed(other.managed), align(other.align),
              allocation_ptr(nullptr), allocation_bytes(0)
            {
            #ifndef NVCC
            if (N > 0)
                {
                allocate();

                std::copy(other.data, other.data+N, data);
                }
            #else
            data = other.data;
            #endif
            }

        //! Assignment operator
        /*! \warn the copy assignment constructor reads from the other array and assumes that array is available on the
                  host. If the GPU isn't synced up, this can lead to erros, so proper multi-GPU synchronization
                  needs to be ensured
         */
        DEVICE ManagedArray& operator=(const ManagedArray<T>& other)
            {
            #ifndef NVCC
            deallocate();
            #endif

            N = other.N;
            managed = other.managed;
            align = other.align;

            #ifndef NVCC
            if (N > 0)
                {
                allocate();

                std::copy(other.data, other.data+N, data);
                }
            #else
            data = other.data;
            #endif

            return *this;
            }

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

        #ifdef ENABLE_CUDA
        //! Attach managed memory to CUDA stream
        void attach_to_stream(cudaStream_t stream) const
            {
            if (managed && data)
                {
                cudaStreamAttachMemAsync(stream, data, 0, cudaMemAttachSingle);
                #if (CUDART_VERSION >= 8000)
                cudaMemAdvise(data, sizeof(T)*N, cudaMemAdviseSetReadMostly, 0);
                #endif
                }
            }
        #endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
            {
            // size in ints (round up)
            unsigned int size_int = (sizeof(T)*N)/sizeof(int);
            if ((sizeof(T)*N) % sizeof(int)) size_int++;

            // align ptr to size of data type
            unsigned long int max_align_bytes = (sizeof(int) > sizeof(T) ? sizeof(int) : sizeof(T))-1;
            char *ptr_align = (char *)(((unsigned long int)ptr + max_align_bytes) & ~max_align_bytes);

            if (size_int*sizeof(int)+max_align_bytes > available_bytes)
                return;

            #if defined (__CUDA_ARCH__)
            // only in GPU code
            unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
            unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

            for (unsigned int cur_offset = 0; cur_offset < size_int; cur_offset += block_size)
                {
                if (cur_offset + tidx < size_int)
                    {
                    ((int *)ptr_align)[cur_offset + tidx] = ((int *)data)[cur_offset + tidx];
                    }
                }

            // make sure all threads have read from data
            __syncthreads();

            // redirect data ptr
            if (tidx == 0)
                {
                data = (T *) ptr_align;
                }

            __syncthreads();
            #endif

            // increment pointer
            ptr = ptr_align + size_int*sizeof(int);
            available_bytes -= size_int*sizeof(int)+max_align_bytes;
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
        #ifndef NVCC
        void allocate()
            {
            data = managed_allocator<T>::allocate_construct_aligned(N, managed, align, allocation_bytes, allocation_ptr);
            }

        void deallocate()
            {
            if (N > 0)
                {
                managed_allocator<T>::deallocate_destroy_aligned(data, N, managed, allocation_ptr);
                }
            }
        #endif

    private:
        mutable T *data;         //!< Data pointer
        unsigned int N;          //!< Number of data elements
        unsigned int managed;    //!< True if we are CUDA managed
        size_t align;            //!< Alignment size
        void *allocation_ptr;    //!< Pointer to un-aligned start of allocation
        size_t allocation_bytes; //!< Total size of allocation, including aligned part
    };
