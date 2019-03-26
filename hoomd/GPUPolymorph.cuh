// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file GPUPolymorph.cuh
 * \brief Defines supporting CUDA functions for GPUPolymorph.
 */

#ifndef HOOMD_GPU_POLYMORPH_CUH_
#define HOOMD_GPU_POLYMORPH_CUH_

#include <type_traits>
#include <cuda_runtime.h>

namespace hoomd
{
namespace gpu
{

//! Method to initialize an object within a kernel
template<class T, typename ...Args>
T* device_new(Args... args);

//! Method to delete an object initialized within a kernel
template<class T>
void device_delete(T* data);

#ifdef NVCC
namespace kernel
{
//! Kernel to initialize and place object into allocated memory.
/*!
 * \tparam T Type of object to initialize.
 * \tparam Args Argument types for constructor of \a T.
 * \param data Allocated device memory to initialize.
 * \param args Argument values to construct \a T.
 * \returns Allocated, initialized pointer to a \a T object.
 *
 * Only one thread executes to avoid race conditions. The object is placed into \a data.
 *
 * \sa hoomd::gpu::device_new
 */
template<class T, typename ...Args>
__global__ void device_construct(void* data, Args... args)
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index != 0) return;
    new (data) T(args...);
    }

template<class T>
__global__ void device_destroy(T* data)
    {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index != 0) return;
    data->~T();
    }
} // end namespace kernel

/*!
 * \tparam T Type of object to initialize.
 * \tparam Args Argument types for constructor of \a T.
 * \param args Argument values to construct \a T.
 * \returns Allocated, initialized pointer to a \a T object.
 *
 * Global memory on the device is first allocated with cudaMalloc. Then, kernel::device_construct is called
 * to initialize and place the object in that memory. Unlike calling new within a kernel, this
 * ensures that the object resides in normal global memory (and not in the limited dynamically allocatable
 * global memory). The device is synchronized to ensure the object is available immediately after the
 * pointer is returned.
 *
 * Note that the \a Args parameter pack has all references removed before forwarding to the kernel. This
 * should ensure that all arguments are passed by copy, even if the user forwards them by reference for
 * efficiency.
 */
template<class T, typename ...Args>
T* device_new(Args... args)
    {
    T* data;
    cudaMalloc((void**)&data, sizeof(T));
    kernel::device_construct<T,typename std::remove_reference<Args>::type...><<<1,1>>>(data, args...);
    cudaDeviceSynchronize();
    return data;
    }

/*!
 * \tparam T Type of object to delete.
 * \param data Object to delete.
 *
 * The destructor for \a data is first called from within a kernel using kernel::device_destroy.
 * In principle, virtual destructors should be chained together if \a T is the base class. (?)
 * After destruction, the memory can be deallocated using cudaFree().
 */
template<class T>
void device_delete(T* data)
    {
    if (data)
        {
        kernel::device_destroy<<<1,1>>>(data);
        cudaDeviceSynchronize();
        cudaFree((void*)data);
        }
    }
#endif // NVCC

} // end namespace gpu
} // end namespace hoomd

#endif // HOOMD_GPU_POLYMORPH_CUH_
