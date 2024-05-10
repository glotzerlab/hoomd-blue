// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
// put a few misc math functions here as they don't have any better home
namespace detail
    {
// !helper to call CPU or GPU signbit
template<class T> HOSTDEVICE inline int signbit(const T& a)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return ::signbit(a);
#else
#ifndef __HIPCC__
    return std::signbit(a);
#else
    return (a >= 0);
#endif
#endif
    }

template<class T> HOSTDEVICE inline T min(const T& a, const T& b)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return ::min(a, b);
#else
#ifndef __HIPCC__
    return std::min(a, b);
#else
    return (a < b) ? a : b;
#endif
#endif
    }

template<class T> HOSTDEVICE inline T max(const T& a, const T& b)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return ::max(a, b);
#else

#ifndef __HIPCC__
    return std::max(a, b);
#else
    return (a > b) ? a : b;
#endif
#endif
    }

template<class T> HOSTDEVICE inline void swap(T& a, T& b)
    {
    T c;
    c = a;
    a = b;
    b = c;
    }

//! A conversion to help unroll function template loops
template<unsigned int> struct int2type
    {
    };

    } // namespace detail
    } // namespace hpmc

    } // namespace hoomd

#undef HOSTDEVICE
