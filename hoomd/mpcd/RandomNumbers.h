// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/RandomNumbers.h
 * \brief Declaration of mpcd::RandomNumbers
 *
 * This header includes templated generators for various types of random
 * numbers required in MPCD. These implementations manipulate uniform numbers
 * drawn by the Saru generator, and work on both the CPU and the GPU.
 */

#ifndef MPCD_RANDOM_NUMBERS_H_
#define MPCD_RANDOM_NUMBERS_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif // NVCC

#define MPCD_2PI 6.283185307179586

namespace mpcd
{
namespace detail
{

//! Generate a random point on the surface of a sphere
template<typename Real>
class SpherePointGenerator
    {
    public:
        DEVICE explicit SpherePointGenerator() {}

        template<typename GeneratorType, typename Real3>
        DEVICE inline void operator()(GeneratorType& rng, Real3& point)
            {
            // draw a random angle
            const Real theta = rng.s(Real(0), Real(MPCD_2PI));

            // draw u (should typically only happen once) ensuring that
            // 1-u^2 > 0 so that the square-root is defined
            Real u, one_minus_u2;
            do
                {
                u = rng.s(Real(-1.0), Real(1.0));
                one_minus_u2 = 1.0f-u*u;
                }
            while (one_minus_u2 < Real(0.0));

            // project onto the sphere surface
            const Real sqrtu = fast::sqrt(one_minus_u2);
            point.x = sqrtu * fast::cos(theta);
            point.y = sqrtu * fast::sin(theta);
            point.z = u;
            }
    };

//! Generate a normally distributed random variable using the Box-Muller method
template<typename Real, bool use_cache=true>
class NormalGenerator
    {
    public:
        DEVICE explicit NormalGenerator()
            : m_cache(false), m_val(0.0)
            {}

        //! Draw a value from the distribution
        /*!
         * \param rng Saru random number generator
         * \returns Normally distributed random variable with mean zero and unit variance
         */
        template<typename GeneratorType>
        DEVICE inline Real operator()(GeneratorType& rng)
            {
            // if available, use value from the cache first
            if (use_cache && m_cache)
                {
                m_cache = false;
                return m_val;
                }

            // draw two uniform random numbers
            const Real u1 = rng.template s<Real>();
            const Real u2 = rng.template s<Real>();

            // apply the Box-Muller transformation
            const Real r = fast::sqrt(Real(-2.0) * fast::log(u1));
            const Real phi = Real(MPCD_2PI) * u2;

            // if enabled, cache the second value in case it is to be reused
            if (use_cache)
                {
                m_val = r * fast::sin(phi);
                m_cache = true;
                }

            return r * fast::cos(phi);
            }

    private:
        bool m_cache;   //!< Flag if cache is valid
        Real m_val;     //!< Value in the cache
    };

//! Generator for gamma-distributed random variables
/*!
 * The probability distribution function is:
 *
 * \f[
 *    p(x) = \frac{1}{\Gamma(\alpha) b^\alpha} x^{\alpha - 1} e^{-x/b}
 * \f]
 *
 * which has parameters \f$\alpha\f$ and \a b. In the Maxwell-Boltzmann
 * distribution, \a x is the kinetic energy of \a N particles, \a b plays the
 * role of the thermal energy \a kT , and \f$\alpha = d(N-1)/2\f$ for
 * dimensionality \a d.
 *
 * The method used to generate the random numbers is a rejection-sampling method:
 *
 *      "A simple method for generating gamma variables", ACM Transactions on
 *      Mathematical Software (TOMS), vol. 26, issue 3, Sept. 2000, 363--372.
 *      https://doi.org/10.1145/358407.358414
 *
 * \tparam Real Precision of the random number
 */
template<typename Real>
class GammaGenerator
    {
    public:
        //! Constructor
        /*!
         * \param alpha
         * \param b
         */
        DEVICE explicit GammaGenerator(const Real alpha, const Real b)
            : m_b(b)
            {
            m_d = alpha - Real(1./3.);
            m_c = fast::rsqrt(m_d)/Real(3.);
            }

        //! Draw a random number from the gamma distribution
        /*!
         * \param rng Saru random number generator
         * \returns A gamma distributed random variate
         *
         * The implementation of this method is inspired by that of the GSL,
         * and also as discussed online:
         *
         *      http://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
         *
         * The squeeze test is performed to bypass some transcendental calls.
         */
        template<typename GeneratorType>
        DEVICE inline Real operator()(GeneratorType& rng)
            {
            Real v;
            while(1)
                {
                // first draw a valid Marsaglia v value using the normal distribution
                Real x;
                do
                    {
                    x = m_normal(rng);
                    v = 1.0f + m_c * x;
                    }
                while (v <= Real(0.));
                v = v*v*v;

                // draw uniform and perform cheap squeeze test first
                const Real x2 = x*x;
                Real u = rng.template s<Real>();
                if (u < 1.0f-0.0331f*x2*x2) break;

                // otherwise, do expensive log comparison
                if (fast::log(u) < 0.5f*x2 + m_d*(1.0f-v+fast::log(v))) break;
                }

            // convert the Gamma(alpha,1) to Gamma(alpha,beta)
            return m_d * v * m_b;
            }

    private:
        Real m_b;       //!< Gamma-distribution b-parameter
        Real m_c;       //!< c-parameter for Marsaglia and Tsang method
        Real m_d;       //!< d-parameter for Marasglia and Tsang method

        NormalGenerator<Real> m_normal; //!< Normal variate generator
    };

} // end namespace detail
} // end namespace mpcd

#undef MPCD_2PI
#endif // #define MPCD_RANDOM_NUMBERS_H_
