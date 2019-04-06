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
#include "RandomNumbers.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
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
        DEVICE inline Saru(uint32_t seed1=0,
                               uint32_t seed2=0,
                               uint32_t counter1=0,
                               uint32_t counter2=0,
                               uint32_t counter3=0);
        //! \name Uniform random numbers in [2**(-65), 1]
        //@{
        //! Draw a random 32-bit unsigned integer
        DEVICE inline uint32_t u32();

        //! Draw a random float in [2**(-65), 1]
        DEVICE inline float f();

        //! Draw a random double in [2**(-65), 1]
        DEVICE inline double d();

        //! Draw a floating-point value in [2**(-65), 1]
        template<class Real>
        DEVICE inline Real s();
        //@}

        //! \name Uniform generators in [a,b]
        //@{
        //! Draw a random float in [a,b]
        DEVICE inline float f(float a, float b);

        //! Draw a random double in [a,b]
        DEVICE inline double d(double a, double b);

        //! Draw a random floating-point value in [a,b]
        template<class Real>
        DEVICE inline Real s(Real a, Real b);
        //@}

        //! \name Other distributions
        //@{
        //! Draw a normally distributed random number (float)
        DEVICE inline float normal(float sigma, float mu=0.0f);

        //! Draw a normally distributed random number (double)
        DEVICE inline double normal(double sigma, double mu=0.0);
        //@}

        // pass through RandomGenerator
        DEVICE inline r123::Philox4x32::ctr_type operator()()
            {
            return m_rng();
            }

    private:
        RandomGenerator m_rng;
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
DEVICE inline Saru::Saru(uint32_t seed1,
                             uint32_t seed2,
                             uint32_t counter1,
                             uint32_t counter2,
                             uint32_t counter3)
    : m_rng(seed1, seed2, counter1, counter2, counter3)
    {
    }

/*!
 * \returns A random uniform 32-bit integer.
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline unsigned int Saru::u32()
    {
    return generate_u32(m_rng);
    }

/*!
 * \returns A random uniform float in [2**(-65), 1].
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline float Saru::f()
    {
    return generate_canonical<float>(m_rng);
    }

/*!
 * \returns A random uniform double in [2**(-65), 1].
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline double Saru::d()
    {
    return generate_canonical<double>(m_rng);
    }

//! Template specialization for float
/*!
 * \returns A random uniform float in in [2**(-65), 1].
 *
 * \post The state of the generator is advanced one step.
 */
template<>
DEVICE inline float Saru::s()
    {
    return generate_canonical<float>(m_rng);
    }

//! Template specialization for double
/*!
 * \returns A random uniform double in in [2**(-65), 1].
 *
 * \post The state of the generator is advanced one step.
 */
template<>
DEVICE inline double Saru::s()
    {
    return generate_canonical<double>(m_rng);
    }

/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b].
 *
 * For most practical purposes, the range returned by this function is [a,b]. This is due to round off error:
 * e.g. for a=1.0, 1.0+2**(-65) == 1.0.
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline float Saru::f(float a, float b)
    {
    UniformDistribution<float> dist(a, b);
    return dist(m_rng);
    }

/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b].
 *
 * For most practical purposes, the range returned by this function is [a,b]. This is due to round off error:
 * e.g. for a=1.0, 1.0+2**(-65) == 1.0.
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline double Saru::d(double a, double b)
    {
    UniformDistribution<double> dist(a, b);
    return dist(m_rng);
    }

//! Template specialization for float
/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b].
 *
 * For most practical purposes, the range returned by this function is [a,b]. This is due to round off error:
 * e.g. for a=1.0, 1.0+2**(-65) == 1.0.
 *
 * \post The state of the generator is advanced one step.
 */
template<>
DEVICE inline float Saru::s(float a, float b)
    {
    UniformDistribution<float> dist(a, b);
    return dist(m_rng);
    }

//! Template specialization for double
/*!
 * \param a Lower bound of range.
 * \param b Upper bound of range.
 * \returns A random uniform float in [a,b].
 *
 * For most practical purposes, the range returned by this function is [a,b]. This is due to round off error:
 * e.g. for a=1.0, 1.0+2**(-65) == 1.0.
 *
 * \post The state of the generator is advanced one step.
 */
template<>
DEVICE inline double Saru::s(double a, double b)
    {
    UniformDistribution<double> dist(a, b);
    return dist(m_rng);
    }

/*!
 * \param sigma Standard deviation
 * \param mu Mean
 *
 * \returns Normally distributed random variable with mean *mu* and standard deviation *sigma*.
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline float Saru::normal(float sigma, float mu)
    {
    NormalDistribution<float> dist(sigma, mu);
    return dist(m_rng);
    }

/*!
 * \param sigma Standard deviation
 * \param mu Mean
 *
 * \returns Normally distributed random variable with mean *mu* and standard deviation *sigma*.
 *
 * \post The state of the generator is advanced one step.
 */
DEVICE inline double Saru::normal(double sigma, double mu)
    {
    NormalDistribution<double> dist(sigma, mu);
    return dist(m_rng);
    }

} // end namespace detail
} // end namespace hoomd

//! Select a random index
/*! \param rng Saru RNG to utilize in the move
    \param max Maximum index to select
    \returns a random number 0 <= i <= max with uniform probability.

    **Method**

    First, round max+1 up to the next nearest power of two -> max2. Then draw random numbers in the range [0 ... max2)
    using 32-but random values and a bitwise and with max2-1. Return the first random number found in the range.


    This method is being kept for support within HPMC kernels that will be updated later on next. Once all code has
    been updated to the new API, this method (and probably all of Saru.h) can be removed.
*/
template <class RNG>
DEVICE inline unsigned int rand_select(RNG& rng, unsigned int max)
    {
    // handle degenerate case where max==0
    if (max == 0)
        return 0;

    // algorithm to round up to the nearest power of two from https://en.wikipedia.org/wiki/Power_of_two
    unsigned int n = max+1;
    n = n - 1;
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    // Note: leaving off the n = n + 1 because we are going to & with next highest power of 2 -1

    unsigned int result;
    do
        {
        result = rng.u32() & n;
        } while(result > max);

    return result;
    }

#undef DEVICE

#endif // HOOMD_SARU_H_
