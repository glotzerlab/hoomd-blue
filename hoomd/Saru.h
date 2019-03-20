// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file hoomd/Saru.h
 * \brief Implementation of the Saru random number generator.
 *
 * This file contains minor modifications and improvements to the original Saru
 * source code made available under the following license:
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

// pull in uint2 type
#include "HOOMDMath.h"

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
 * Saru is a pseudo-random number generator that requires only a 64-bit state vector.
 * The generator is first seeded using one, two, or three 32-bit unsigned integers.
 * The seeds are hashed together to generate an initially random state consisting
 * of two 32-bit words. The hash routines pass TestU01's Crush. The seeded generator
 * is then able to generate streams of random numbers using a combination of
 * a linear congruential generator (LCG) and an Offset Weyl Sequence (OWS) to advance
 * the state. The streaming generator has a period of 3666320093*2^32, and passes
 * DIEHARD, Rabbit, Gorilla, and TestU01's SmallCrush, Crush, and BigCrush. On
 * the GPU, typical use is then to seed the generator per-kernel and per-thread
 * to generate random microstreams (e.g., each thread gets a generator hashed from
 * the particle tag, the timestep, and a user-defined seed).
 *
 * See
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
        //! Default constructor
        HOSTDEVICE inline Saru();
        //! One-seed constructor
        HOSTDEVICE inline Saru(unsigned int seed);
        //! Two-seed constructor
        HOSTDEVICE inline Saru(unsigned int seed1, unsigned int seed2);
        //! Three-seed constructor
        HOSTDEVICE inline Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3);

        //! Run-time computed advancement of the state of the generator
        HOSTDEVICE inline void advance(unsigned int steps);

        //! Efficient compile-time advancement of the generator
        /*!
         * \tparam steps Number of steps to advance.
         *
         * The state of the generator is advanced \a steps. This operation is
         * very efficient because it is done through template tricks at compile-time.
         */
        template <unsigned int steps>
        HOSTDEVICE inline void advance()
            {
            advanceWeyl<steps>(); advanceLCG<steps>();
            }

        //! Efficient compile-time rewind of the state of the generator.
        /*!
         * \tparam steps Number of steps to advance.
         *
         * The state of the generator is rewound \a steps. This operation is
         * very efficient because it is done through template tricks at compile-time.
         */
        template <unsigned int steps>
        HOSTDEVICE inline void rewind()
            {
            /*
             * OK to advance negative steps in LCG, it's done mod 2^32 so it's
             * the same as advancing 2^32-1 steps, which is correct!
             */
            rewindWeyl<steps>(); advanceLCG<-steps>();
            }

        //! Fork the state of the generator
        template <unsigned int seed>
        HOSTDEVICE Saru fork() const;

        //! \name Base generators
        //@{
        //! Draw a random 32-bit unsigned integer advanced ahead \a steps
        template <unsigned int steps>
        HOSTDEVICE inline unsigned int u32();

        //! Draw a random float on [0,1) advanced ahead \a steps
        template <unsigned int steps>
        HOSTDEVICE inline float f();

        //! Draw a random double on [0,1) advanced ahead \a steps
        template <unsigned int steps>
        HOSTDEVICE inline double d();

        //! Draw a random float on [a,b) advanced ahead \a steps
        template <unsigned int steps>
        HOSTDEVICE inline float f(float low, float b);

        //! Draw a random double on [a,b) advanced ahead \a steps
        template <unsigned int steps>
        HOSTDEVICE inline double d(double low, double b);
        //@}

        //! \name Single-step advancers
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

        //! \name Methods for stdlib compatibility.
        //@{
        typedef unsigned int result_type;
        //! Wrapper to u32() call.
        result_type operator()();

        //! Smallest number that can be returned by u32. Set as 0x00.
        static constexpr result_type min() { return 0; }

        //! Largest number that can be returned by u32. Set as 0xffffffff.
        static constexpr result_type max() { return 0xffffffff; } // Max for 32-bit numbers (TODO: use STD limits.h ?)
        //@}

    private:
        uint2 state;    //!< Internal state of the generator

        //! \name Internal advancement methods
        //@{
        static const unsigned int LCGA=0x4beb5d59; //!< Full period 32 bit LCG
        static const unsigned int LCGC=0x2600e1f7;
        static const unsigned int oWeylPeriod=0xda879add; //!< Prime period 3666320093
        static const unsigned int oWeylOffset=0x8009d14b;
        static const unsigned int oWeylDelta=(oWeylPeriod-0x80000000)+(oWeylOffset-0x80000000); //!< wraps mod 2^32

        //! Advance the Linear Congruential Generator
        template <unsigned int steps>
        HOSTDEVICE inline  void advanceLCG();

        //! Advance any Offset Weyl Sequence
        template <unsigned int offset, unsigned int delta, unsigned int modulus, unsigned int steps>
        HOSTDEVICE inline unsigned int advanceAnyWeyl(unsigned int x);

        //! Advance the Saru Offset Weyl Sequence
        template <unsigned int steps>
        HOSTDEVICE inline void advanceWeyl();

        //! Rewind the Saru Offset Weyl Sequence
        template <unsigned int steps>
        HOSTDEVICE inline void rewindWeyl();
        //@}

        //! \name Advancement helper metaprograms
        //@{
        //! Helper to compute A^N mod 2^32
        template<unsigned int A, unsigned int N>
        struct CTpow
            {
            static const unsigned int value=(N&1?A:1)*CTpow<A*A, N/2>::value;
            };
        //! Template specialization to terminate recursion, A^0 = 1
        template<unsigned int A>
        struct CTpow<A, 0>
            {
            static const unsigned int value=1;
            };

        //! Helper to compute the power series 1+A+A^2+A^3+A^4+A^5..+A^(N-1) mod 2^32
        /*!
         * Based on recursion:
         * \verbatim
         * g(A,n)= (1+A)*g(A*A, n/2);      if n is even
         * g(A,n)= 1+A*(1+A)*g(A*A, n/2);  if n is ODD (since n/2 truncates)
         * \endverbatim
         */
        template<unsigned int A, unsigned int N>
        struct CTpowseries
            {
            static const unsigned int recurse=(1+A)*CTpowseries<A*A, N/2>::value;
            static const unsigned int value=  (N&1) ? 1+A*recurse : recurse;
            };
        //! Template specialization for N=0, gives 0
        template<unsigned int A>
        struct CTpowseries<A, 0>
            {
            static const unsigned int value=0;
            };
        //! Template specialization for N=0, gives 1
        template<unsigned int A>
        struct CTpowseries<A, 1>
            {
            static const unsigned int value=1;
            };

        //! Helper to compute A*B mod m.  Tricky only because of implicit 2^32 modulus.
        /*!
         * Based on recursion:
         *
         * \verbatim
         * if A is even, then A*B mod m =  (A/2)*(B+B mod m) mod m.
         * if A is odd,  then A*B mod m =  (B+((A/2)*(B+B mod m) mod m)) mod m.
         * \endverbatim
         */
        template <unsigned int A, unsigned int B, unsigned int m>
        struct CTmultmod
            {
            // (A/2)*(B*2) mod m
            static const unsigned int temp=  CTmultmod< A/2, (B>=m-B ? B+B-m : B+B), m>::value;
            static const unsigned int value= A&1 ? ((B>=m-temp) ? B+temp-m: B+temp) : temp;
            };
        //! Template specialization to terminate recursion
        template <unsigned int B, unsigned int m>
        struct CTmultmod<0, B, m>
            {
            static const unsigned int value=0;
            };
        //@}
    };

/*!
 * The default constructor initializes a simple dummy state.
 */
HOSTDEVICE inline Saru::Saru()
    {
    state.x=0x12345678; state.y=12345678;
    }

/*!
 * \param seed Seed to the generator.
 *
 * This seeding was carefully tested for good churning with 1, 2, and 3 bit flips.
 * All 32 incrementing counters (each of the circular shifts) pass the TestU01 Crush tests.
 */
HOSTDEVICE inline Saru::Saru(unsigned int seed)
    {
    state.x = 0x79dedea3*(seed^(((signed int)seed)>>14));
    state.y = seed ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    }

/*!
 * \param seed1 First seed.
 * \param seed2 Second seed.
 *
 * One bit of entropy is lost by mixing because the input seeds have 64 bits,
 * but after mixing there are only 63 left.
 */
HOSTDEVICE inline Saru::Saru(unsigned int seed1, unsigned int seed2)
    {
    seed2 += seed1<<16;
    seed1 += seed2<<11;
    seed2 += ((signed int)seed1)>>7;
    seed1 ^= ((signed int)seed2)>>3;
    seed2 *= 0xA5366B4D;
    seed2 ^= seed2>>10;
    seed2 ^= ((signed int)seed2)>>19;
    seed1 += seed2^0x6d2d4e11;

    state.x = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    state.y = (state.x+seed2) ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    }

/*!
 * \param seed1 First seed.
 * \param seed2 Second seed.
 * \param seed3 Third seed.
 *
 * The seeds are premixed before dropping to 64 bits.
 */
HOSTDEVICE inline Saru::Saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
    {
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;

    state.x = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    state.y = (state.x+seed2) ^ (((signed int)state.x)>>8);
    state.x = state.x + (state.y*(state.y^0xdddf97f5));
    state.y = 0xABCB96F7 + (state.y>>1);
    }

/*!
 * \param Number of steps to advance the generator.
 *
 * This run-time method is less efficient than compile-time advancement, but is
 * still pretty fast.
 */
HOSTDEVICE inline void Saru::advance(unsigned int steps)
    {
    // Computes the LCG advancement AND the Weyl D*E mod m simultaneously
    unsigned int currentA=LCGA;
    unsigned int currentC=LCGC;

    unsigned int currentDelta=oWeylDelta;
    unsigned int netDelta=0;

    while (steps)
        {
        if (steps&1)
            {
            state.x=currentA*state.x+currentC; // LCG step
            if (netDelta<oWeylPeriod-currentDelta) netDelta+=currentDelta;
            else netDelta+=currentDelta-oWeylPeriod;
            }

        // Change the LCG to step at twice the rate as before
        currentC+=currentA*currentC;
        currentA*=currentA;

        // Change the Weyl delta to step at 2X rate
        if (currentDelta<oWeylPeriod-currentDelta) currentDelta+=currentDelta;
        else currentDelta+=currentDelta-oWeylPeriod;

        steps/=2;
        }

    // Apply the net delta to the Weyl state.
    if (state.y-oWeylOffset<oWeylPeriod-netDelta) state.y+=netDelta;
    else state.y+=netDelta-oWeylPeriod;
    }

/*!
 * \tparam seed Seed for creating the new generator.
 * \returns A new instance of the generator.
 *
 * The user-supplied seed is bitchurned at compile time, which is very efficient.
 * Churning takes small user values like 1 2 3 and hashes them to become roughly
 * uncorrelated.
 */
template <unsigned int seed>
HOSTDEVICE Saru Saru::fork() const
    {
    const unsigned int churned1=0xDEADBEEF ^ (0x1fc4ce47*(seed^(seed>>13)));
    const unsigned int churned2=0x1234567+(0x82948463*(churned1^(churned1>>20)));
    const unsigned int churned3=0x87654321^(0x87655677*(churned2^(churned2>>16)));

    Saru z;
    z.state.x=churned2+state.x+(churned3^state.y);
    unsigned int add=(z.state.x+churned1)>>1;
    if (z.state.y-oWeylOffset<oWeylPeriod-add) z.state.y+=add;
    else z.state.y+=add-oWeylPeriod;
    return z;
    }

/*!
 * \tparam Number of steps to advance.
 * \returns A random 32-bit unsigned integer.
 *
 * \post The state of the generator is advanced \a steps.
 *
 * This method implements the heart of the Saru number generator. The state is
 * advance by \a steps, and is then bitswizzled to return a random integer. This
 * simple generation method has been shown to unconditionally pass a battery of
 * tests of randomness.
 */
template <unsigned int steps>
HOSTDEVICE inline unsigned int Saru::u32()
    {
    advanceLCG<steps>();
    advanceWeyl<steps>();
    const unsigned int v=(state.x ^ (state.x>>26))+state.y;
    return (v^(v>>20))*0x6957f5a7;
    }

/*!
 * \tparam Number of steps to advance.
 * \returns A random uniform float in [0,1).
 *
 * \post The state of the generator is advanced \a steps.
 *
 * Floats have 23-bits of mantissa. The values here are generated using a
 * conversion to signed int, followed by a shift, which is usually more optimized
 * by the compiler.
 */
template <unsigned int steps>
HOSTDEVICE inline float Saru::f()
    {
    const float TWO_N32 = 2.32830643653869628906250e-10f; // 2^-32
    return ((signed int)u32<steps>())*TWO_N32+0.5f;
    }

/*!
 * \tparam Number of steps to advance.
 * \returns A random uniform double in [0,1).
 *
 * \post The state of the generator is advanced \a steps.
 *
 * Doubles have 52-bits of mantissa. 32-bits are drawn from a random integer from
 * the generator, while the remaining 20-bits are taken from the current state.
 * These low 20-bits are less random, but are quick to draw. The double is generated
 * using the method described in:
 *
 * J. A. Doornik, "Conversion of High-Period Random Numbers to Floating Point",
 * ACM Transactions on Modeling and Computer Simulation (TOMACS) 17, 3 (2007).
 * https://doi.org/10.1145/1189756.1189759.
 *
 * However, the variates are \b not shifted so that the interval is [0,1) rather
 * than (0,1). This is done for sake of consistency with the C++11 <random> and
 * GSL implementations of their random number generators.
 */
template <unsigned int steps>
HOSTDEVICE inline double Saru::d()
    {
    const double TWO_N32 = 2.32830643653869628906250e-10; // 2^-32
    const double TWO_N52 = 2.22044604925031308084726e-16; // 2^-52
    return ((signed int)u32<steps>())*TWO_N32+((signed int)(state.x & 0x000fffff))*TWO_N52+0.5;
    }

/*!
 * \tparam Number of steps to advance.
 * \returns A random uniform float in [a,b).
 *
 * \post The state of the generator is advanced \a steps.
 */
template <unsigned int steps>
HOSTDEVICE inline float Saru::f(float a, float b)
    {
    return a + (b-a)*f<steps>();
    }

/*!
 * \tparam Number of steps to advance.
 * \returns A random uniform double in [a,b).
 *
 * \post The state of the generator is advanced \a steps.
 */
template <unsigned int steps>
HOSTDEVICE inline double Saru::d(double a, double b)
    {
    return a + (b-a)*d<steps>();
    }

/*!
 * \returns A random uniform 32-bit integer.
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline unsigned int Saru::u32()
    {
    return u32<1>();
    }

/*!
 * \returns A random uniform float in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline float Saru::f()
    {
    return f<1>();
    }

/*!
 * \returns A random uniform double in [0,1).
 *
 * \post The state of the generator is advanced one step.
 */
HOSTDEVICE inline double Saru::d()
    {
    return d<1>();
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
    return f<1>(a, b);
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
    return d<1>(a, b);
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

/*!
 * \tparam steps Number of steps to advance.
 */
template <unsigned int steps>
HOSTDEVICE inline void Saru::advanceLCG()
    {
    state.x = CTpow<LCGA,steps>::value*state.x+LCGC*CTpowseries<LCGA,steps>::value;
    }

/*!
 * \param x
 *
 * \tparam offset
 * \tparam delta
 * \tparam modulus
 * \tparam steps Number of steps to advance.
 */
template <unsigned int offset, unsigned int delta, unsigned int modulus, unsigned int steps>
HOSTDEVICE inline unsigned int Saru::advanceAnyWeyl(unsigned int x)
    {
    const unsigned int fullDelta=CTmultmod<delta, steps%modulus, modulus>::value;
    /* the runtime code boils down to this single constant-filled line. */
    return x+((x-offset>modulus-fullDelta) ? fullDelta-modulus : fullDelta);
    }

/*!
 * \tparam steps Number of steps to advance.
 */
template <unsigned int steps>
HOSTDEVICE inline void Saru::advanceWeyl()
    {
    state.y = advanceAnyWeyl<oWeylOffset,oWeylDelta,oWeylPeriod,steps>(state.y);
    }

//! Partial template specialization when only advancing one step
template <>
HOSTDEVICE inline void Saru::advanceWeyl<1>()
    {
    state.y = state.y+oWeylOffset+((((signed int)state.y)>>31)&oWeylPeriod);
    }

/*
 * \tparam steps Number of steps to rewind.
 */
template <unsigned int steps>
HOSTDEVICE inline void Saru::rewindWeyl()
    {
    state.y = advanceAnyWeyl<oWeylOffset,oWeylPeriod-oWeylDelta,oWeylPeriod,steps>(state.y);
    }

} // end namespace detail
} // end namespace hoomd

#undef HOSTDEVICE

#endif // HOOMD_SARU_H_
