// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#ifndef NVCC
#include "managed_allocator.h"

#include <algorithm>
#endif

#undef DEVICE
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
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
        DEVICE T& operator[](unsigned int i)
            {
            return data[i];
            }

        //! random access operator (const version)
        DEVICE const T& operator[](unsigned int i) const
            {
            return data[i];
            }

        //! Get pointer to array data
        DEVICE T * get()
            {
            return data;
            }

        //! Get pointer to array data (const version)
        DEVICE const T* get() const
            {
            return data;
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
        T *data;            //!< Data pointer
        unsigned int N;     //!< Number of data elements
        unsigned int managed;       //!< True if we are CUDA managed
    };
