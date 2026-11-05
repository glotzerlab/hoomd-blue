// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_VECTOR_TYPES_H_
#define HOOMD_VECTOR_TYPES_H_

/*
 * This file provides built-in vector types for CPU-only builds. The data types
 * and alignments match those specified in the CUDA Programming Guide. It should
 * NEVER be included in GPU-enabled builds because it may create multiple,
 * conflicting definitions of these data types due to differences in annotations.
 * In GPU-enabled builds, include <cuda_runtime.h> to get access to these data
 * types, or just include HOOMDMath.h.
 */
#ifdef ENABLE_GPU
#error Use CUDA built-in vector types in GPU-enabled builds instead.
#endif

/*
 * signed char
 */

struct alignas(1) char1
    {
    signed char x;
    };

struct alignas(2) char2
    {
    signed char x;
    signed char y;
    };

struct alignas(1) char3
    {
    signed char x;
    signed char y;
    signed char z;
    };

struct alignas(4) char4
    {
    signed char x;
    signed char y;
    signed char z;
    signed char w;
    };

inline char1 make_char1(signed char x)
    {
    return char1 {x};
    }

inline char2 make_char2(signed char x, signed char y)
    {
    return char2 {x, y};
    }

inline char3 make_char3(signed char x, signed char y, signed char z)
    {
    return char3 {x, y, z};
    }

inline char4 make_char4(signed char x, signed char y, signed char z, signed char w)
    {
    return char4 {x, y, z, w};
    }

/*
 * unsigned char
 */

struct alignas(1) uchar1
    {
    unsigned char x;
    };

struct alignas(2) uchar2
    {
    unsigned char x;
    unsigned char y;
    };

struct alignas(1) uchar3
    {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    };

struct alignas(4) uchar4
    {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
    };

inline uchar1 make_uchar1(unsigned char x)
    {
    return uchar1 {x};
    }

inline uchar2 make_uchar2(unsigned char x, unsigned char y)
    {
    return uchar2 {x, y};
    }

inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
    {
    return uchar3 {x, y, z};
    }

inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
    {
    return uchar4 {x, y, z, w};
    }

/*
 * signed int
 */
struct alignas(4) int1
    {
    signed int x;
    };

struct alignas(8) int2
    {
    signed int x;
    signed int y;
    };

struct alignas(4) int3
    {
    signed int x;
    signed int y;
    signed int z;
    };

struct alignas(16) int4
    {
    signed int x;
    signed int y;
    signed int z;
    signed int w;
    };

inline int1 make_int1(signed int x)
    {
    return int1 {x};
    }

inline int2 make_int2(signed int x, signed int y)
    {
    return int2 {x, y};
    }

inline int3 make_int3(signed int x, signed int y, signed int z)
    {
    return int3 {x, y, z};
    }

inline int4 make_int4(signed int x, signed int y, signed int z, signed int w)
    {
    return int4 {x, y, z, w};
    }

/*
 * unsigned int
 */
struct alignas(4) uint1
    {
    unsigned int x;
    };

struct alignas(8) uint2
    {
    unsigned int x;
    unsigned int y;
    };

struct alignas(4) uint3
    {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    };

struct alignas(16) uint4
    {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
    };

inline uint1 make_uint1(unsigned int x)
    {
    return uint1 {x};
    }

inline uint2 make_uint2(unsigned int x, unsigned int y)
    {
    return uint2 {x, y};
    }

inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
    {
    return uint3 {x, y, z};
    }

inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
    {
    return uint4 {x, y, z, w};
    }

/*
 * float
 */
struct alignas(4) float1
    {
    float x;
    };

struct alignas(8) float2
    {
    float x;
    float y;
    };

struct alignas(4) float3
    {
    float x;
    float y;
    float z;
    };

struct alignas(16) float4
    {
    float x;
    float y;
    float z;
    float w;
    };

inline float1 make_float1(float x)
    {
    return float1 {x};
    }

inline float2 make_float2(float x, float y)
    {
    return float2 {x, y};
    }

inline float3 make_float3(float x, float y, float z)
    {
    return float3 {x, y, z};
    }

inline float4 make_float4(float x, float y, float z, float w)
    {
    return float4 {x, y, z, w};
    }

/*
 * double
 */
struct alignas(8) double1
    {
    double x;
    };

struct alignas(16) double2
    {
    double x;
    double y;
    };

struct alignas(8) double3
    {
    double x;
    double y;
    double z;
    };

struct alignas(16) double4_16a
    {
    double x;
    double y;
    double z;
    double w;
    };

struct alignas(32) double4_32a
    {
    double x;
    double y;
    double z;
    double w;
    };

// double4 is equivalent to double4_16a by specification, but it was deprecated
// in CUDA 13.0. It is included as a deprecated alias.
using double4 [[deprecated("Use double4_16a instead.")]] = double4_16a;

inline double1 make_double1(double x)
    {
    return double1 {x};
    }

inline double2 make_double2(double x, double y)
    {
    return double2 {x, y};
    }

inline double3 make_double3(double x, double y, double z)
    {
    return double3 {x, y, z};
    }

inline double4_16a make_double4_16a(double x, double y, double z, double w)
    {
    return double4_16a {x, y, z, w};
    }

inline double4_32a make_double4_32a(double x, double y, double z, double w)
    {
    return double4_32a {x, y, z, w};
    }

[[deprecated("Use make_double4_16a instead.")]]
inline double4 make_double4(double x, double y, double z, double w)
    {
    return double4 {x, y, z, w};
    }

#endif // HOOMD_VECTOR_TYPES_H_
