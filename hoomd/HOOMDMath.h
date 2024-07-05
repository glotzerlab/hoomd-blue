// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __HOOMD_MATH_H__
#define __HOOMD_MATH_H__

/*! \file HOOMDMath.h
    \brief Common setup include for all hoomd math operations
*/

// for vector types
#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#else

// for builds on systems where CUDA is not available, include copies of the CUDA header
// files which define the vector types (float4, etc...)
#include "hoomd/extern/cudacpu_vector_functions.h"
#include "hoomd/extern/cudacpu_vector_types.h"
#endif

// bring in math.h
#ifndef __HIPCC__

// define HOOMD_LLVMJIT_BUILD to prevent the need for python and pybind includes
// this simplifies LLVM code generation
#ifndef HOOMD_LLVMJIT_BUILD
// include python.h first to silence _XOPEN_SOURCE redefinition warnings
#include <Python.h>
#include <pybind11/pybind11.h>
#endif

#include <cmath>
#include <math.h>
#endif

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
// HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
#ifndef ENABLE_HIP
//! Define complex type
typedef float2 hipfftComplex;
//! Double complex type
typedef double2 hipfftDoubleComplex;
#endif

#if HOOMD_LONGREAL_SIZE == 32
//! Floating point type (single precision)
typedef float Scalar;
//! Floating point type with x,y elements (single precision)
typedef float2 Scalar2;
//! Floating point type with x,y elements (single precision)
typedef float3 Scalar3;
//! Floating point type with x,y,z,w elements (single precision)
typedef float4 Scalar4;

typedef float LongReal;
typedef float2 LongReal2;
typedef float3 LongReal3;
typedef float4 LongReal4;

#elif HOOMD_LONGREAL_SIZE == 64
//! Floating point type (double precision)
typedef double Scalar;
//! Floating point type with x,y elements (double precision)
typedef double2 Scalar2;
//! Floating point type with x,y,z elements (double precision)
typedef double3 Scalar3;
//! Floating point type with x,y,z,w elements (double precision)
typedef double4 Scalar4;

typedef double LongReal;
typedef double2 LongReal2;
typedef double3 LongReal3;
typedef double4 LongReal4;

#else
#error HOOMD_LONGREAL_SIZE must be 32 or 64.
#endif

#if HOOMD_SHORTREAL_SIZE == 32
typedef float ShortReal;
typedef float2 ShortReal2;
typedef float3 ShortReal3;
typedef float4 ShortReal4;
#elif HOOMD_SHORTREAL_SIZE == 64
typedef double ShortReal;
typedef double2 ShortReal2;
typedef double3 ShortReal3;
typedef double4 ShortReal4;
#else
#error HOOMD_SHORTREAL_SIZE must be 32 or 64.
#endif

//! make a scalar2 value
HOSTDEVICE inline Scalar2 make_scalar2(Scalar x, Scalar y)
    {
    Scalar2 retval;
    retval.x = x;
    retval.y = y;
    return retval;
    }

//! make a scalar3 value
HOSTDEVICE inline Scalar3 make_scalar3(Scalar x, Scalar y, Scalar z)
    {
    Scalar3 retval;
    retval.x = x;
    retval.y = y;
    retval.z = z;
    return retval;
    }

//! make a scalar4 value
HOSTDEVICE inline Scalar4 make_scalar4(Scalar x, Scalar y, Scalar z, Scalar w)
    {
    Scalar4 retval;
    retval.x = x;
    retval.y = y;
    retval.z = z;
    retval.w = w;
    return retval;
    }

#ifndef __HIPCC__
//! Stuff an integer inside a float
HOSTDEVICE inline float __int_as_float(int a)
    {
        union {
        int a;
        float b;
        } u;

    u.a = a;

    return u.b;
    }
#endif // __HIPCC__

//! Stuff an integer inside a double
HOSTDEVICE inline double __int_as_double(int a)
    {
        union {
        int a;
        double b;
        } u;

    // make sure it is not uninitialized
    u.b = 0.0;
    u.a = a;

    return u.b;
    }

//! Stuff an integer inside a Scalar
HOSTDEVICE inline Scalar __int_as_scalar(int a)
    {
        union {
        int a;
        Scalar b;
        } u;

    // make sure it is not uninitialized
    u.b = Scalar(0.0);
    u.a = a;

    return u.b;
    }

#ifndef __HIPCC__
//! Extract an integer from a float stuffed by __int_as_float()
HOSTDEVICE inline int __float_as_int(float b)
    {
        union {
        int a;
        float b;
        } u;

    u.b = b;

    return u.a;
    }
#endif // __HIPCC__

//! Extract an integer from a double stuffed by __int_as_double()
HOSTDEVICE inline int __double_as_int(double b)
    {
        union {
        int a;
        double b;
        } u;

    u.b = b;

    return u.a;
    }

//! Extract an integer from a Scalar stuffed by __int_as_scalar()
HOSTDEVICE inline int __scalar_as_int(Scalar b)
    {
        union {
        int a;
        Scalar b;
        } u;

    u.b = b;

    return u.a;
    }

//! Export relevant hoomd math functions to python
#ifndef __HIPCC__
#ifndef HOOMD_LLVMJIT_BUILD
namespace detail
    {
void export_hoomd_math_functions(pybind11::module& m);
    }
#endif
#endif

//! Fastmath routines
/*! Routines in the fast namespace map to fast math routines on the CPU and GPU. Where possible,
   these use the less accurate intrinsics on the GPU (i.e. __sinf). The routines are provide
   overloads for both single and double so that macro tricks aren't needed to handle single and
   double precision code.
*/
namespace fast
    {
//! Compute the reciprocal square root of x
inline HOSTDEVICE float rsqrt(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
#ifdef __HIP_PLATFORM_NVCC__
    return ::rsqrtf(x);
#elif defined(__HIP_PLATFORM_HCC__)
    return ::__frsqrt_rn(x);
#endif
#else
    return 1.0f / ::sqrtf(x);
#endif
    }

//! Compute the reciprocal square root of x
inline HOSTDEVICE double rsqrt(double x)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_NVCC__)
    return ::rsqrt(x);
#else
    return 1.0 / ::sqrt(x);
#endif
    }

//! Compute the sin of x
inline HOSTDEVICE float sin(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return __sinf(x);
#else
    return ::sinf(x);
#endif
    }

//! Compute the sin of x
inline HOSTDEVICE double sin(double x)
    {
    return ::sin(x);
    }

//! Compute the cos of x
inline HOSTDEVICE float cos(float x)
    {
#if __HIP_DEVICE_COMPILE__
    return __cosf(x);
#else
    return ::cosf(x);
#endif
    }

//! Compute the cos of x
inline HOSTDEVICE double cos(double x)
    {
    return ::cos(x);
    }

//! Compute both of sin of x and cos of x with float precision
inline HOSTDEVICE void sincos(float x, float& s, float& c)
    {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__APPLE__)
    __sincosf(x, &s, &c);
#else
    ::sincosf(x, &s, &c);
#endif
    }

//! Compute both of sin of x and cos of x with double precision
inline HOSTDEVICE void sincos(double x, double& s, double& c)
    {
#if defined(__HIP_DEVICE_COMPILE__)
    ::sincos(x, &s, &c);
#elif defined(__APPLE__)
    ::__sincos(x, &s, &c);
#else
    ::sincos(x, &s, &c);
#endif
    }

//! Compute both of sin of x and cos of PI * x with float precision
inline HOSTDEVICE void sincospi(float x, float& s, float& c)
    {
#if defined(__HIP_DEVICE_COMPILE__)
    ::sincospif(x, &s, &c);
#elif defined(__APPLE__)
    __sincospif(x, &s, &c);
#else
    fast::sincos(float(M_PI) * x, s, c);
#endif
    }

//! Compute both of sin of x and cos of x with double precision
inline HOSTDEVICE void sincospi(double x, double& s, double& c)
    {
#if defined(__HIP_DEVICE_COMPILE__)
    ::sincospi(x, &s, &c);
#elif defined(__APPLE__)
    ::__sincospi(x, &s, &c);
#else
    fast::sincos(M_PI * x, s, c);
#endif
    }

//! Compute the pow of x,y with single precison via exp(log) refactoring - NOTE: UNDEFINED FOR
//! NEGATIVE BASES
inline HOSTDEVICE float pow(float x, float y)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return __expf(y * __logf(x));
#else
    return ::expf(y * logf(x));
#endif
    }

//! Compute the pow of x,y with double precision via exp(log) refactoring - NOTE: UNDEFINED FOR
//! NEGATIVE BASES
inline HOSTDEVICE double pow(double x, double y)
    {
    return ::exp(y * log(x));
    }

//! Compute the exp of x
inline HOSTDEVICE float exp(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return __expf(x);
#else
    return ::expf(x);
#endif
    }

//! Compute the exp of x
inline HOSTDEVICE double exp(double x)
    {
    return ::exp(x);
    }

//! Compute the natural log of x
inline HOSTDEVICE float log(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return __logf(x);
#else
    return ::logf(x);
#endif
    }

//! Compute the natural log of x
inline HOSTDEVICE double log(double x)
    {
    return ::log(x);
    }

//! Compute the sqrt of x
inline HOSTDEVICE float sqrt(float x)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_HCC__)
    return ::__fsqrt_rn(x);
#else
    return ::sqrtf(x);
#endif
    }

//! Compute the sqrt of x
inline HOSTDEVICE double sqrt(double x)
    {
    return ::sqrt(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE float erfc(float x)
    {
    return ::erfcf(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE double erfc(double x)
    {
    return ::erfc(x);
    }

//! Compute the acos of x
inline HOSTDEVICE float acos(float x)
    {
    return ::acosf(x);
    }

//! Compute the acos of x
inline HOSTDEVICE double acos(double x)
    {
    return ::acos(x);
    }
    } // namespace fast

//! Maximum accuracy math routines
/*! Routines in the slow namespace map to the most accurate version of the math routines on the CPU
   and GPU. The routines are provide overloads for both single and double so that macro tricks
   aren't needed to handle single and double precision code.

    These routines are intended to be used e.g. in integrators, where numerical stability is most
   important.
*/
namespace slow
    {
//! Compute the reciprocal square root of x
inline HOSTDEVICE float rsqrt(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return ::rsqrtf(x);
#else
    return 1.0f / ::sqrtf(x);
#endif
    }

//! Compute the reciprocal square root of x
inline HOSTDEVICE double rsqrt(double x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return ::rsqrt(x);
#else
    return 1.0 / ::sqrt(x);
#endif
    }

//! Compute the sin of x
inline HOSTDEVICE float sin(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return sinf(x);
#else
    return ::sinf(x);
#endif
    }

//! Compute the sin of x
inline HOSTDEVICE double sin(double x)
    {
    return ::sin(x);
    }

//! Compute the cos of x
inline HOSTDEVICE float cos(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return cosf(x);
#else
    return ::cosf(x);
#endif
    }

//! Compute the cos of x
inline HOSTDEVICE double cos(double x)
    {
    return ::cos(x);
    }

//! Compute the tan of x
inline HOSTDEVICE float tan(float x)
    {
    return ::tanf(x);
    }

//! Compute the tan of x
inline HOSTDEVICE double tan(double x)
    {
    return ::tan(x);
    }

//! Compute the pow of x,y
inline HOSTDEVICE float pow(float x, float y)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return powf(x, y);
#else
    return ::powf(x, y);
#endif
    }

//! Compute the sin of x
inline HOSTDEVICE double pow(double x, double y)
    {
    return ::pow(x, y);
    }

//! Compute the exp of x
inline HOSTDEVICE float exp(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return expf(x);
#else
    return ::expf(x);
#endif
    }

//! Compute the exp of x
inline HOSTDEVICE double exp(double x)
    {
    return ::exp(x);
    }

//! Compute the natural log of x
inline HOSTDEVICE float log(float x)
    {
#ifdef __HIP_DEVICE_COMPILE__
    return logf(x);
#else
    return ::logf(x);
#endif
    }

//! Compute the natural log of x
inline HOSTDEVICE double log(double x)
    {
    return ::log(x);
    }

//! Compute the sqrt of x
inline HOSTDEVICE float sqrt(float x)
    {
    return ::sqrtf(x);
    }

//! Compute the sqrt of x
inline HOSTDEVICE double sqrt(double x)
    {
    return ::sqrt(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE float erfc(float x)
    {
    return ::erfcf(x);
    }

//! Compute the erfc of x
inline HOSTDEVICE double erfc(double x)
    {
    return ::erfc(x);
    }

//! Compute the acos of x
inline HOSTDEVICE float acos(float x)
    {
    return ::acosf(x);
    }

//! Compute the acos of x
inline HOSTDEVICE double acos(double x)
    {
    return ::acos(x);
    }

//! Compute the atan of x
inline HOSTDEVICE float atan(float x)
    {
    return ::atanf(x);
    }

//! Compute the atan of x
inline HOSTDEVICE double atan(double x)
    {
    return ::atan(x);
    }

//! Compute the floor of x
inline HOSTDEVICE float floor(float x)
    {
    return ::floorf(x);
    }

//! Compute the floor of x
inline HOSTDEVICE double floor(double x)
    {
    return ::floor(x);
    }

/// Compute the hypberbolic tangent of x
inline HOSTDEVICE double tanh(double x)
    {
    return ::tanh(x);
    }

/// Compute the hypberbolic tangent of x
inline HOSTDEVICE float tanh(float x)
    {
    return ::tanhf(x);
    }

/// Compute the rint of x
inline HOSTDEVICE double rint(double x)
    {
    return ::rint(x);
    }

/// Compute the rint of x
inline HOSTDEVICE float rint(float x)
    {
    return ::rintf(x);
    }
    } // namespace slow

    } // end namespace hoomd

// ------------ Vector math functions --------------------------
//! Comparison operator needed for export of std::vector<uint2>
HOSTDEVICE inline bool operator==(const uint2& a, const uint2& b)
    {
    return (a.x == b.x && a.y == b.y);
    }

//! Comparison operator needed for export of std::vector<Scalar3>
HOSTDEVICE inline bool operator==(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
    }

//! Comparison operator needed for export of std::vector<Scalar3>
HOSTDEVICE inline bool operator!=(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return !(a == b);
    }

//! Comparison operator needed for export of std::vector<Scalar4>
HOSTDEVICE inline bool operator==(const hoomd::Scalar4& a, const hoomd::Scalar4& b)
    {
    return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
    }

//! Comparison operator needed for export of std::vector<Scalar4>
HOSTDEVICE inline bool operator!=(const hoomd::Scalar4& a, const hoomd::Scalar4& b)
    {
    return !(a == b);
    }

//! Vector addition
HOSTDEVICE inline hoomd::Scalar3 operator+(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

#if !defined(ENABLE_HIP) || defined(__HIP_PLATFORM_NVCC__)
//! Vector addition
HOSTDEVICE inline hoomd::Scalar3& operator+=(hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
    }
#endif

//! Vector subtraction
HOSTDEVICE inline hoomd::Scalar3 operator-(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
//! Vector subtraction
HOSTDEVICE inline hoomd::Scalar3& operator-=(hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
    }

//! Vector multiplication (component-wise)
HOSTDEVICE inline hoomd::Scalar3 operator*(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

//! Vector multiplication
HOSTDEVICE inline hoomd::Scalar3& operator*=(hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
    }

//! Vector division (component-wise)
HOSTDEVICE inline hoomd::Scalar3 operator/(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a.x / b.x, a.y / b.y, a.z / b.z);
    }
//! Scalar - vector multiplication
HOSTDEVICE inline hoomd::Scalar3 operator*(const hoomd::Scalar& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a * b.x, a * b.y, a * b.z);
    }
//! Scalar - vector multiplication
HOSTDEVICE inline hoomd::Scalar3 operator*(const hoomd::Scalar3& a, const hoomd::Scalar& b)
    {
    return hoomd::make_scalar3(a.x * b, a.y * b, a.z * b);
    }
//! Vector - scalar multiplication
HOSTDEVICE inline hoomd::Scalar3& operator*=(hoomd::Scalar3& a, const hoomd::Scalar& b)
    {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
    }
//! Vector - scalar division
HOSTDEVICE inline hoomd::Scalar3 operator/(const hoomd::Scalar3& a, const hoomd::Scalar& b)
    {
    return hoomd::make_scalar3(a.x / b, a.y / b, a.z / b);
    }
//! Vector - scalar division in place
HOSTDEVICE inline hoomd::Scalar3& operator/=(hoomd::Scalar3& a, const hoomd::Scalar& b)
    {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
    }
//! Vector - scalar division
HOSTDEVICE inline hoomd::Scalar3 operator/(const hoomd::Scalar& a, const hoomd::Scalar3& b)
    {
    return hoomd::make_scalar3(a / b.x, a / b.y, a / b.z);
    }
//! Vector unary -
HOSTDEVICE inline hoomd::Scalar3 operator-(const hoomd::Scalar3& a)
    {
    return hoomd::make_scalar3(-a.x, -a.y, -a.z);
    }
//! Vector dot product
HOSTDEVICE inline hoomd::Scalar dot(const hoomd::Scalar3& a, const hoomd::Scalar3& b)
    {
    return a.x * b.x + a.y * b.y + a.z * b.z;
    }

// ----------- Integer vector math functions ----------------------
//! Integer vector addition
HOSTDEVICE inline int3 operator+(const int3& a, const int3& b)
    {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
//! Integer vector unary addition
HOSTDEVICE inline int3 operator+=(int3& a, const int3& b)
    {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
    }
//! Integer vector subtraction
HOSTDEVICE inline int3 operator-(const int3& a, const int3& b)
    {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
    }
//! Integer vector unary subtraction
HOSTDEVICE inline int3 operator-=(int3& a, const int3& b)
    {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
    }
//! Integer vector unary -
HOSTDEVICE inline int3 operator-(const int3& a)
    {
    return make_int3(-a.x, -a.y, -a.z);
    }
//! Integer vector comparison
HOSTDEVICE inline bool operator==(const int3& a, const int3& b)
    {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
    }
//! Integer vector comparison
HOSTDEVICE inline bool operator!=(const int3& a, const int3& b)
    {
    return (a.x != b.x || a.y != b.y || a.z != b.z);
    }

// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE
#undef DEVICE

#endif // __HOOMD_MATH_H__
