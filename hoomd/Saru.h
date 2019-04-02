// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*!
 * \file hoomd/Saru.h
 * \brief random123 with the Saru API.
 *
 * This file contains a complete reimplementation of the Saru API based on random123
 * The original Saru source code made available under the following license:
 *
 * \verbatim
 * Copyright (c) 2008 Steve Worley < m a t h g e e k@(my last name).com >
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 * \endverbatim
 */

#ifndef HOOMD_SARU_H_
#define HOOMD_SARU_H_

#include "HOOMDMath.h"

#ifdef ENABLE_CUDA
// ensure that curand is included before random123. This avoids multiple defiintion issues
// unfortunately, at the cost of random123 using the coefficients provided by curand
// for now, they are the same
#include <curand_kernel.h>
#endif

#include <math.h>
#include <hoomd/extern/random123/include/Random123/philox.h>
#include <type_traits>

namespace r123 {
// from random123/examples/uniform.hpp
using std::make_signed;
using std::make_unsigned;

#if defined(__CUDACC__) || defined(_LIBCPP_HAS_NO_CONSTEXPR)
// Amazing! cuda thinks numeric_limits::max() is a __host__ function, so
// we can't use it in a device function.
//
// The LIBCPP_HAS_NO_CONSTEXP test catches situations where the libc++
// library thinks that the compiler doesn't support constexpr, but we
// think it does.  As a consequence, the library declares
// numeric_limits::max without constexpr.  This workaround should only
// affect a narrow range of compiler/library pairings.
//
// In both cases, we find max() by computing ~(unsigned)0 right-shifted
// by is_signed.
template <typename T>
R123_CONSTEXPR R123_STATIC_INLINE R123_CUDA_DEVICE T maxTvalue()
    {
    typedef typename make_unsigned<T>::type uT;
    return (~uT(0)) >> std::numeric_limits<T>::is_signed;
    }
#else
template <typename T>
R123_CONSTEXPR R123_STATIC_INLINE T maxTvalue()
    {
    return std::numeric_limits<T>::max();
    }
#endif

// u01: Input is a W-bit integer (signed or unsigned).  It is cast to
//   a W-bit unsigned integer, multiplied by Ftype(2^-W) and added to
//   Ftype(2^(-W-1)).  A good compiler should optimize it down to an
//   int-to-float conversion followed by a multiply and an add, which
//   might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  result is a uniformly distributed floating point number in [0, 1].
//  The result is never exactly 0.0.
//  The smallest value returned is 2^-W.
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0.
template <typename Ftype, typename Itype>
R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01(Itype in)
    {
    typedef typename make_unsigned<Itype>::type Utype;
    R123_CONSTEXPR Ftype factor = Ftype(1.)/(maxTvalue<Utype>() + Ftype(1.));
    R123_CONSTEXPR Ftype halffactor = Ftype(0.5)*factor;
#if R123_UNIFORM_FLOAT_STORE
    volatile Ftype x = Utype(in)*factor; return x+halffactor;
#else
    return Utype(in)*factor + halffactor;
#endif
    }

// uneg11: Input is a W-bit integer (signed or unsigned).  It is cast
//    to a W-bit signed integer, multiplied by Ftype(2^-(W-1)) and
//    then added to Ftype(2^(-W-2)).  A good compiler should optimize
//    it down to an int-to-float conversion followed by a multiply and
//    an add, which might be fused, depending on the architecture.
//
//  If the input is a uniformly distributed integer, then the
//  output is a uniformly distributed floating point number in [-1, 1].
//  The result is never exactly 0.0.
//  The smallest absolute value returned is 2^-(W-1)
//  Let M be the number of mantissa bits in Ftype.
//  If W>M  then the largest value retured is 1.0 and the smallest is -1.0.
//  If W<=M then the largest value returned is the largest Ftype less than 1.0
//    and the smallest value returned is the smallest Ftype greater than -1.0.
template <typename Ftype, typename Itype>
R123_CUDA_DEVICE R123_STATIC_INLINE Ftype uneg11(Itype in)
    {
    typedef typename make_signed<Itype>::type Stype;
    R123_CONSTEXPR Ftype factor = Ftype(1.)/(maxTvalue<Stype>() + Ftype(1.));
    R123_CONSTEXPR Ftype halffactor = Ftype(0.5)*factor;
#if R123_UNIFORM_FLOAT_STORE
    volatile Ftype x = Stype(in)*factor; return x+halffactor;
#else
    return Stype(in)*factor + halffactor;
#endif
    }

// u01fixedpt:  Return a "fixed point" number in (0,1).  Let:
//   W = width of Itype, e.g., 32 or 64, regardless of signedness.
//   M = mantissa bits of Ftype, e.g., 24, 53 or 64
//   B = min(M, W)
// Then the 2^(B-1) possible output values are:
//    2^-B*{1, 3, 5, ..., 2^B - 1}
// The smallest output is: 2^-B
// The largest output is:  1 - 2^-B
// The output is never exactly 0.0, nor 0.5, nor 1.0.
// The 2^(B-1) possible outputs:
//   - are equally likely,
//   - are uniformly spaced by 2^-(B-1),
//   - are balanced around 0.5
template <typename Ftype, typename Itype>
R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01fixedpt(Itype in)
    {
    typedef typename make_unsigned<Itype>::type Utype;
    R123_CONSTEXPR int excess = std::numeric_limits<Utype>::digits - std::numeric_limits<Ftype>::digits;
    if(excess>=0)
        {
        R123_CONSTEXPR int ex_nowarn = (excess>=0) ? excess : 0;
        R123_CONSTEXPR Ftype factor = Ftype(1.)/(Ftype(1.) + ((maxTvalue<Utype>()>>ex_nowarn)));
        return (1 | (Utype(in)>>ex_nowarn)) * factor;
        }
    else
        return u01<Ftype>(in);
    }

// from random123/examples/boxmuller.hpp

/*
 * take two 32bit unsigned random values and return a float2 with
 * two random floats in a normal distribution via a Box-Muller transform
 */
R123_CUDA_DEVICE R123_STATIC_INLINE float2 boxmuller(uint32_t u0, uint32_t u1)
    {
    float r;
    float2 f;
    fast::sincospi(uneg11<float>(u0), f.x, f.y);
    r = sqrtf(-2.f * logf(u01<float>(u1))); // u01 is guaranteed to avoid 0.
    f.x *= r;
    f.y *= r;
    return f;
    }

/*
 * take two 64bit unsigned random values and return a double2 with
 * two random doubles in a normal distribution via a Box-Muller transform
 */
R123_CUDA_DEVICE R123_STATIC_INLINE double2 boxmuller(uint64_t u0, uint64_t u1)
    {
    double r;
    double2 f;

    fast::sincospi(uneg11<double>(u0), f.x, f.y);
    r = sqrt(-2. * log(u01<double>(u1))); // u01 is guaranteed to avoid 0.
    f.x *= r;
    f.y *= r;
    return f;
    }
}

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif // NVCC

namespace hoomd
{
namespace detail
{

//! Saru random number generator
/*!
 * random123 is a counter based random number generator. Given an input seed vector,
 * it produces a random output. Outputs from one seed to the next are not correlated.
 * This class implements a convenience API around random123 that allows short streams
 * (less than 2**32-1) of random numbers starting from a seed of up to 5 uint32_t values.
 * The convenience API is based on a random number generator called Saru that was
 * previously used in HOOMD.
 *
 * Internally, we use the philox 4x32 RNG from random123, The first two seeds map to the
 * key and the remaining seeds map to the counter. One element from the counter is used
 * to generate the stream of values. Constructors provide ways to conveniently initialize
 * the RNG with any number of seeds or counters. Warning! All constructors with fewer
 * than 5 input values are equivalent to the 5-input constructor with 0's for the
 * values not specified.
 *
 * Counter based RNGs are useful for MD simulations: See
 *
 * C.L. Phillips, J.A. Anderson, and S.C. Glotzer. "Pseudo-random number generation
 * for Brownian Dynamics and Dissipative Particle Dynamics simulations on GPU devices",
 * J. Comput. Phys. 230, 7191-7201 (2011).
 *
 * and
 *
 * Y. Afshar, F. Schmid, A. Pishevar, and S. Worley. "Exploiting seeding of random
 * number generators for efficient domain decomposition parallelization of dissipative
 * particle dynamics", Comput. Phys. Commun. 184, 1119-1128 (2013).
 *
 * for more details.
 */
class Saru
    {
    public:
        //! Five-value constructor
        HOSTDEVICE inline Saru(unsigned int seed1=0,
                               unsigned int seed2=0,
                               unsigned int counter1=0,
                               unsigned int counter2=0,
                               unsigned int counter3=0);
        //! \name Uniform random numbers
        //@{
        //! Draw a random 32-bit unsigned integer
        HOSTDEVICE inline unsigned int u32();

        //! Draw a random float on [0,1)
        HOSTDEVICE inline float f();

        //! Draw a random double on [0,1)
        HOSTDEVICE inline double d();

        //! Draw a floating-point value on [0,1)
        template<class Real>
        HOSTDEVICE inline Real s();
        //@}

        //! \name Uniform generators on [a,b)
        //@{
        //! Draw a random float in [a,b)
        HOSTDEVICE inline float f(float a, float b);

        //! Draw a random double in [a,b)
        HOSTDEVICE inline double d(double a, double b);

        //! Draw a random floating-point value in [a,b)
        template<class Real>
        HOSTDEVICE inline Real s(Real a, Real b);
        //@}

        //! \name Other distributions
        //@{
        //! Draw a normal random number
        template<class Real>
        HOSTDEVICE inline Real normal();
        //@}

    private:
        r123::Philox4x32::key_type m_key;   //!< RNG key
        r123::Philox4x32::ctr_type m_ctr;   //!< RNG counter

    };

/*!
 * \param seed1 First seed.
 * \param seed2 Second seed.
 * \param counter1 First counter.
 * \param counter2 Second counter
 * \param counter3 Third counter
 *
 * Initialize the random number stream with two seeds and one counter. Seeds and counters are somewhat interchangeable.
 * Seeds should be more static (i.e. user seed, RNG id) while counters should be more dynamic (i.e. particle tag).
 */
HOSTDEVICE inline Saru::Saru(unsigned int seed1,
                             unsigned int seed2,
                             unsigned int counter1,
                             unsigned int counter2,
                             unsigned int counter3)
    {
    m_key = {{seed1, seed2}};
    m_ctr = {{0, counter3, counter2, counter1}};
    }

/*!
 * \returns A random uniform 32-bit integer.
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline unsigned int Saru::u32()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr[0] += 1;
    return u[0];
    }

/*!
 * \returns A random uniform float in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline float Saru::f()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr[0] += 1;
    return r123::u01<float>(u[0]);
    }

/*!
 * \returns A random uniform double in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline double Saru::d()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr[0] += 1;
    uint64_t u64 = uint64_t(u[0]) << 32 | u[1];
    return r123::u01<double>(u64);
    }

//! Template specialization for float
/*!
 * \returns A random uniform float in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline float Saru::s()
    {
    return f();
    }

//! Template specialization for double
/*!
 * \returns A random uniform double in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline double Saru::s()
    {
    return d();
    }

/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline float Saru::f(float a, float b)
    {
    return a + (b-a)*f();
    }

/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform double in [a,b).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline double Saru::d(double a, double b)
    {
    return a + (b-a)*d();
    }

//! Template specialization for float
/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b).
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline float Saru::s(float a, float b)
    {
    return f(a, b);
    }

//! Template specialization for double
/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform double in [a,b).
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline double Saru::s(double a, double b)
    {
    return d(a, b);
    }

//! Normal distribution
/*!
 * \returns Normally distributed random variable with mean zero and unit variance
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline float Saru::normal()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr[0] += 1;
    float2 n = r123::boxmuller(u[0], u[1]);
    return n.x;
    // note: If there is a need, we could add an API that returns two normally distributed numbers for no extra cost
    }

//! Normal distribution
/*!
 * \returns Normally distributed random variable with mean zero and unit variance
 *
 * \post The state of the generator is advanced one step.
 */
template<>
HOSTDEVICE inline double Saru::normal()
    {
    r123::Philox4x32 rng;
    r123::Philox4x32::ctr_type u = rng(m_ctr, m_key);
    m_ctr[0] += 1;
    uint64_t u64_0 = uint64_t(u[0]) << 32 | u[1];
    uint64_t u64_1 = uint64_t(u[2]) << 32 | u[3];
    double2 n = r123::boxmuller(u64_0, u64_1);
    return n.x;
    // note: If there is a need, we could add an API that returns two normally distributed numbers for no extra cost
    }

} // end namespace detail
} // end namespace hoomd

#undef HOSTDEVICE

#endif // HOOMD_SARU_H_
