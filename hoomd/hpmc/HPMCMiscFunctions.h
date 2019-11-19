// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


# pragma once

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace hpmc
{

// put a few misc math functions here as they don't have any better home
namespace detail
    {
    // !helper to call CPU or GPU signbit
    template <class T> HOSTDEVICE inline int signbit(const T& a)
        {
        #ifdef __HIP_DEVICE_COMPILE__
        return ::signbit(a);
        #else
        return std::signbit(a);
        #endif
        }

    template <class T> HOSTDEVICE inline T min(const T& a, const T& b)
        {
        #ifdef __HIP_DEVICE_COMPILE__
        return ::min(a,b);
        #else
        return std::min(a,b);
        #endif
        }

    template <class T> HOSTDEVICE inline T max(const T& a, const T& b)
        {
        #ifdef __HIP_DEVICE_COMPILE__
        return ::max(a,b);
        #else
        return std::max(a,b);
        #endif
        }

    template<class T> HOSTDEVICE inline void swap(T& a, T&b)
        {
        T c;
        c = a;
        a = b;
        b = c;
        }
    }

} // end namespace hpmc

#undef HOSTDEVICE
