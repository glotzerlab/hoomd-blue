// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef NVCC
#include "managed_allocator.h"

#include <algorithm>
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
            #ifndef NVCC
            deallocate();
            #endif

            N = other.N;
            managed = other.managed;

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

        HOSTDEVICE void load_shared(char *& ptr, bool load=true) const
            {
            // align to size of data type
            //uintptr_t addr =  (uintptr_t)ptr;
            //if (addr % sizeof(T) != 0) addr += sizeof(T) - addr % sizeof(T);
            //ptr = (char *)addr;

            #if defined (__CUDA_ARCH__)
            if (load)
                {
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

                __syncthreads();

                // redirect data ptr
                data = (T *) ptr;
                }
            #endif

            // increment pointer
            ptr += N*sizeof(T);
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
                managed_allocator<T>::deallocate(data, N, managed);
                }
            }
        #endif

    private:
        mutable T *data;       //!< Data pointer
        unsigned int N;        //!< Number of data elements
        unsigned int managed;  //!< True if we are CUDA managed
    };
