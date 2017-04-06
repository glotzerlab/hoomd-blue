// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef NVCC
#include "managed_allocator.h"

#include <algorithm>
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
            : data(nullptr), N(0), managed(0)
            { }

        #ifndef NVCC
        ManagedArray(unsigned int _N, bool _managed)
            : N(_N), managed(_managed)
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
        DEVICE ManagedArray(const ManagedArray<T>& other)
            : N(other.N), managed(other.managed)
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
        DEVICE ManagedArray& operator=(const ManagedArray<T>& other)
            {
            printf("Deallocating N==%d\n",N);
            #ifndef NVCC
            deallocate();
            #endif

            printf("N==%d\n",other.N);
            N = other.N;
            managed = other.managed;

            #ifndef NVCC
            if (N > 0)
                {
                allocate();
                printf("before std::copy %p %p\n",other.data,data);
                std::copy(other.data, other.data+N, data);
                printf("after std::copy\n");
                }
            #else
            data = other.data;
            #endif

            return *this;
            }

        //! random access operator
        HOSTDEVICE T& operator[](unsigned int i)
            {
            return data[i];
            }

        //! random access operator (const version)
        HOSTDEVICE const T& operator[](unsigned int i) const
            {
            return data[i];
            }

        //! Get pointer to array data
        HOSTDEVICE T * get()
            {
            return data;
            }

        //! Get pointer to array data (const version)
        HOSTDEVICE const T* get() const
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
            // we use at most this many bytes due to aligment
            unsigned int max_bytes = sizeof(T) -1 + sizeof(T)*N;

            if (max_bytes > available_bytes) return;

            available_bytes -= max_bytes;

            #if defined (__CUDA_ARCH__)
            // only in GPU code

            // align ptr to size of data type
            ptr = (char *)((unsigned long int)(ptr + (sizeof(T) - 1)) & ~((unsigned long int) sizeof(T) - 1));

            unsigned int size_int = (sizeof(T)*N)/sizeof(int);

            unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
            unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;

            for (unsigned int cur_offset = 0; cur_offset < size_int; cur_offset += block_size)
                {
                if (cur_offset + tidx < size_int)
                    {
                    ((int *)ptr)[cur_offset + tidx] = ((int *)data)[cur_offset + tidx];
                    }
                }

            // make sure all threads have read from data
            __syncthreads();

            // redirect data ptr
            data = (T *) ptr;

            // increment pointer
            ptr += N*sizeof(T);
            #endif
            }

    protected:
        #ifndef NVCC
        void allocate()
            {
            data = managed_allocator<T>::allocate(N, managed);
            }

        void deallocate()
            {
            if (N > 0)
                {
                printf("Destroying %d\n",N);
                managed_allocator<T>::deallocate(data, N, managed);
                printf("Done.\n");   
                }
            }
        #endif

    private:
        mutable T *data;       //!< Data pointer
        unsigned int N;        //!< Number of data elements
        unsigned int managed;  //!< True if we are CUDA managed
    };
