// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HOOMDMath.h"
#include "VectorMath.h"

#ifndef __HIPCC__
#include <algorithm>
#endif

#ifndef __AABB_H__
#define __AABB_H__

/*! \file AABB.h
    \brief Basic AABB routines
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE

#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE __attribute__((always_inline))
#endif

#if !defined(__HIPCC__) && defined(__SSE__)
#include <immintrin.h>
#endif

namespace hoomd
    {
namespace detail
    {
#if defined(__SSE__) && !defined(__HIPCC__)
inline __m128 sse_load_vec3_float(const vec3<float>& value)
    {
    float in[4];
    in[0] = value.x;
    in[1] = value.y;
    in[2] = value.z;
    in[3] = 0.0f;
    return _mm_loadu_ps(in);
    }

inline vec3<float> sse_unload_vec3_float(const __m128& v)
    {
    float out[4];
    _mm_storeu_ps(out, v);
    return vec3<float>(out[0], out[1], out[2]);
    }

inline __m128 sse_load_quat_float(const quat<float>& value)
    {
    float in[4];
    in[0] = value.v.x;
    in[1] = value.v.y;
    in[2] = value.v.z;
    in[3] = value.s;
    return _mm_loadu_ps(in);
    }

inline quat<float> sse_unload_quat_float(const __m128& v)
    {
    float out[4];
    _mm_storeu_ps(out, v);
    return quat<float>(out[3], vec3<float>(out[0], out[1], out[2]));
    }
#endif

#if defined(__AVX__) && !defined(__HIPCC__)
inline __m256d sse_load_vec3_double(const vec3<double>& value)
    {
    double in[4];
    in[0] = value.x;
    in[1] = value.y;
    in[2] = value.z;
    in[3] = 0.0;
    return _mm256_loadu_pd(in);
    }

inline vec3<double> sse_unload_vec3_double(const __m256d& v)
    {
    double out[4];
    _mm256_storeu_pd(out, v);
    return vec3<double>(out[0], out[1], out[2]);
    }

inline __m256d sse_load_quat_double(const quat<double>& value)
    {
    double in[4];
    in[0] = value.x;
    in[1] = value.y;
    in[2] = value.z;
    in[3] = value.s;
    return _mm256_loadu_pd(in);
    }

inline quat<double> sse_unload_quat_double(const __m256d& v)
    {
    double out[4];
    _mm256_storeu_pd(out, v);
    return quat<float>(out[3], vec3<float>(out[0], out[1], out[2]));
    }
#endif

//! Axis aligned bounding box
/*! An AABB represents a bounding volume defined by an axis-aligned bounding box. It is stored as
   plain old data with a lower and upper bound. This is to make the most common operation of AABB
   overlap testing fast.

    Do not access data members directly. AABB uses SSE and AVX optimizations and the internal data
   format changes. It also changes between the CPU and GPU. Instead, use the accessor methods
   getLower(), getUpper() and getPosition().

    Operations are provided as free functions to perform the following operations:

    - merge()
    - overlap()
    - contains()
*/
struct
#ifndef __HIPCC__
    __attribute__((visibility("default")))
#endif
    AABB
    {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
    __m256d lower_v; //!< Lower left corner (AVX data type)
    __m256d upper_v; //!< Upper left corner (AVX data type)

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
    __m128 lower_v; //!< Lower left corner (SSE data type)
    __m128 upper_v; //!< Upper left corner (SSE data type)

#else
    quat<Scalar> lower; //!< Lower left corner
    quat<Scalar> upper; //!< Upper right corner

#endif

    unsigned int tag; //!< Optional tag id, useful for particle ids

    //! Default construct a 0 AABB
    DEVICE AABB() : tag(0)
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        double in = 0.0f;
        lower_v = _mm256_broadcast_sd(&in);
        upper_v = _mm256_broadcast_sd(&in);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        float in = 0.0f;
        lower_v = _mm_load_ps1(&in);
        upper_v = _mm_load_ps1(&in);

#endif
        // vec3 constructors zero themselves
        }

    //! Construct an AABB from the given lower and upper corners
    /*! \param _lower Lower left corner of the AABB
        \param _upper Upper right corner of the AABB
    */
    DEVICE AABB(const vec3<Scalar>& _lower, const vec3<Scalar>& _upper) : tag(0)
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        lower_v = sse_load_vec3_double(_lower);
        upper_v = sse_load_vec3_double(_upper);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        lower_v = sse_load_vec3_float(_lower);
        upper_v = sse_load_vec3_float(_upper);

#else
        lower = quat<Scalar>(0,_lower);
        upper = quat<Scalar>(0,_upper);

#endif
        }

    //! Construct an AABB from a sphere
    /*! \param _position Position of the sphere
        \param radius Radius of the sphere
    */
    DEVICE AABB(const vec3<Scalar>& _position, Scalar radius) : tag(0)
        {
        vec3<Scalar> new_lower, new_upper;
        new_lower.x = _position.x - radius;
        new_lower.y = _position.y - radius;
        new_lower.z = _position.z - radius;
        new_upper.x = _position.x + radius;
        new_upper.y = _position.y + radius;
        new_upper.z = _position.z + radius;

#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        lower_v = sse_load_vec3_double(new_lower);
        upper_v = sse_load_vec3_double(new_upper);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        lower_v = sse_load_vec3_float(new_lower);
        upper_v = sse_load_vec3_float(new_upper);

#else
        lower.v = new_lower;
        upper.v = new_upper;
	lower.s = upper.s = 0.0;

#endif
        }


     //! Construct a four-dimensional AABB from a hypersphere
    /*! \param _position Position of the hypersphere
        \param radius Radius of the sphere
    */
    DEVICE AABB(const quat<Scalar>& _position, Scalar radius) : tag(0)
        {
        quat<Scalar> new_lower, new_upper;
        new_lower.v.x = _position.v.x - radius;
        new_lower.v.y = _position.v.y - radius;
        new_lower.v.z = _position.v.z - radius;
        new_lower.s = _position.s - radius;
        new_upper.v.x = _position.v.x + radius;
        new_upper.v.y = _position.v.y + radius;
        new_upper.v.z = _position.v.z + radius;
        new_upper.s = _position.s + radius;

        #if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(NVCC) && 0
        lower_v = sse_load_quat_double(new_lower);
        upper_v = sse_load_quat_double(new_upper);

        #elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(NVCC)
        lower_v = sse_load_quat_float(new_lower);
        upper_v = sse_load_quat_float(new_upper);

        #else
         lower = new_lower;
         upper = new_upper;

        #endif
        }

    //! Construct an AABB from a point with a particle tag
    /*! \param _position Position of the point
        \param _tag Global particle tag id
    */
    DEVICE AABB(const vec3<Scalar>& _position, unsigned int _tag) : tag(_tag)
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        lower_v = sse_load_vec3_double(_position);
        upper_v = sse_load_vec3_double(_position);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        lower_v = sse_load_vec3_float(_position);
        upper_v = sse_load_vec3_float(_position);

#else
        lower.v = _position;
        upper.v = _position;
	lower.s = upper.s = 0.0;

#endif
        }

    //! Get the AABB's position
    DEVICE vec3<Scalar> getPosition() const
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        double half = 0.5;
        __m256d half_v = _mm256_broadcast_sd(&half);
        __m256d pos_v = _mm256_mul_pd(half_v, _mm256_add_pd(lower_v, upper_v));
        return sse_unload_vec3_double(pos_v);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        float half = 0.5f;
        __m128 half_v = _mm_load_ps1(&half);
        __m128 pos_v = _mm_mul_ps(half_v, _mm_add_ps(lower_v, upper_v));
        return sse_unload_vec3_float(pos_v);

#else
        return (lower.v + upper.v) / Scalar(2);

#endif
        }

    //! Get the AABB's position (4d)
    DEVICE quat<Scalar> getPositionS() const
        {
        #if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(NVCC) && 0
        double half = 0.5;
        __m256d half_v = _mm256_broadcast_sd(&half);
        __m256d pos_v = _mm256_mul_pd(half_v, _mm256_add_pd(lower_v, upper_v));
        return sse_unload_quat_double(pos_v);

        #elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(NVCC)
        float half = 0.5f;
        __m128 half_v = _mm_load_ps1(&half);
        __m128 pos_v = _mm_mul_ps(half_v, _mm_add_ps(lower_v, upper_v));
        return sse_unload_quat_float(pos_v);

        #else
        return quat<Scalar>((lower.s+upper.s)/Scalar(2.0),(lower.v + upper.v) / Scalar(2));

         #endif
         }

    //! Get the AABB's lower point
    DEVICE vec3<Scalar> getLower() const
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        return sse_unload_vec3_double(lower_v);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        return sse_unload_vec3_float(lower_v);

#else
        return lower.v;

#endif
        }

    //! Get the AABB's upper point
    DEVICE vec3<Scalar> getUpper() const
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        return sse_unload_vec3_double(upper_v);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        return sse_unload_vec3_float(upper_v);

#else
        return upper.v;

#endif
        }


    //! Get the AABB's lower point (Sphere)
    DEVICE quat<Scalar> getLowerS() const
        {
        #if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(NVCC) && 0
        return sse_unload_quat_double(lower_v);

        #elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(NVCC)
        return sse_unload_quat_float(lower_v);

        #else
        return lower;

        #endif
        }

    //! Get the AABB's upper point (Sphere)
    DEVICE quat<Scalar> getUpperS() const
        {
        #if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(NVCC) && 0
        return sse_unload_quat_double(upper_v);

        #elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(NVCC)
        return sse_unload_quat_float(upper_v);

        #else
         return upper;

         #endif
	}


    //! Translate the AABB by the given vector
    DEVICE void translate(const vec3<Scalar>& v)
        {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
        __m256d v_v = sse_load_vec3_double(v);
        lower_v = _mm256_add_pd(lower_v, v_v);
        upper_v = _mm256_add_pd(upper_v, v_v);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
        __m128 v_v = sse_load_vec3_float(v);
        lower_v = _mm_add_ps(lower_v, v_v);
        upper_v = _mm_add_ps(upper_v, v_v);

#else
        upper.v += v;
        lower.v += v;

#endif
        }
    }
#ifndef HOOMD_LLVMJIT_BUILD
    __attribute__((aligned(32)));
#else
    ;
#endif

//! Check if two AABBs overlap
/*! \param a First AABB
    \param b Second AABB
    \returns true when the two AABBs overlap, false otherwise
*/
DEVICE inline bool overlap(const AABB& a, const AABB& b)
    {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
    int r0 = _mm256_movemask_pd(_mm256_cmp_pd(b.upper_v, a.lower_v, 0x11)); // 0x11=lt
    int r1 = _mm256_movemask_pd(_mm256_cmp_pd(b.lower_v, a.upper_v, 0x1e)); // 0x1e=gt
    return !(r0 || r1);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
    int r0 = _mm_movemask_ps(_mm_cmplt_ps(b.upper_v, a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmpgt_ps(b.lower_v, a.upper_v));
    return !(r0 || r1);

#else
    return !(b.upper.v.x < a.lower.v.x || b.lower.v.x > a.upper.v.x || b.upper.v.y < a.lower.v.y
             || b.lower.v.y > a.upper.v.y || b.upper.v.z < a.lower.v.z || b.lower.v.z > a.upper.v.z
	     || b.upper.s < a.lower.s || b.lower.s > a.upper.s);

#endif
    }

//! Check if one AABB contains another
/*! \param a First AABB
    \param b Second AABB
    \returns true when b is fully contained within a
*/
DEVICE inline bool contains(const AABB& a, const AABB& b)
    {
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
    int r0 = _mm256_movemask_pd(_mm256_cmp_pd(b.lower_v, a.lower_v, 0x1d)); // 0x1d=ge
    int r1 = _mm256_movemask_pd(_mm256_cmp_pd(b.upper_v, a.upper_v, 0x12)); // 0x12=le
    return ((r0 & r1) == 0xF);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
    int r0 = _mm_movemask_ps(_mm_cmpge_ps(b.lower_v, a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmple_ps(b.upper_v, a.upper_v));
    return ((r0 & r1) == 0xF);

#else
    return (b.lower.v.x >= a.lower.v.x && b.upper.v.x <= a.upper.v.x && b.lower.v.y >= a.lower.v.y
            && b.upper.v.y <= a.upper.v.y && b.lower.v.z >= a.lower.v.z && b.upper.v.z <= a.upper.v.z
	    && b.lower.s >= a.lower.s && b.upper.s <= a.upper.s);

#endif
    }

#ifndef __HIPCC__
//! Merge two AABBs
/*! \param a First AABB
    \param b Second AABB
    \returns A new AABB that encloses *a* and *b*
*/
DEVICE inline AABB merge(const AABB& a, const AABB& b)
    {
    AABB new_aabb;
#if defined(__AVX__) && !defined(SINGLE_PRECISION) && !defined(__HIPCC__) && 0
    new_aabb.lower_v = _mm256_min_pd(a.lower_v, b.lower_v);
    new_aabb.upper_v = _mm256_max_pd(a.upper_v, b.upper_v);

#elif defined(__SSE__) && defined(SINGLE_PRECISION) && !defined(__HIPCC__)
    new_aabb.lower_v = _mm_min_ps(a.lower_v, b.lower_v);
    new_aabb.upper_v = _mm_max_ps(a.upper_v, b.upper_v);

#else
    new_aabb.lower.v.x = std::min(a.lower.v.x, b.lower.v.x);
    new_aabb.lower.v.y = std::min(a.lower.v.y, b.lower.v.y);
    new_aabb.lower.v.z = std::min(a.lower.v.z, b.lower.v.z);
    new_aabb.lower.s = std::min(a.lower.s, b.lower.s);
    new_aabb.upper.v.x = std::max(a.upper.v.x, b.upper.v.x);
    new_aabb.upper.v.y = std::max(a.upper.v.y, b.upper.v.y);
    new_aabb.upper.v.z = std::max(a.upper.v.z, b.upper.v.z);
    new_aabb.upper.s = std::max(a.upper.s, b.upper.s);

#endif

    return new_aabb;
    }
#endif

// end group overlap
/*! @}*/

    }; // end namespace detail

    }; // end namespace hoomd

#undef DEVICE
#endif //__AABB_H__
