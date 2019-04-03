// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "hoomd/mpcd/RandomNumbers.h"
#include "hoomd/Saru.h"
#include <vector>
#include <iostream>
#include <iomanip>

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN()

//! Test case for SpherePointGenerator
/*!
 * When drawing uniformly on a sphere, the pdf should satisfy:
 *
 * \f[
 *    \int f(\omega) d\omega = 1 = \int d\theta d\phi f(\theta, \phi).
 * \f]
 *
 * The proper distribution satisfying this is:
 *
 * \f[
 *   f(\theta, \phi) = sin(\phi) / 4\pi
 * \f]
 *
 * because d\omega = sin(\phi) d\theta d\phi.
 *
 * The marginal probability of each spherical coordinate is then
 * \f[
 *   f(\theta) = 1/2\pi \\
 *   f(\phi) = sin(\phi)/2
 * \f]
 */
UP_TEST( sphere_point_test )
    {
    // initialize the histograms
    const double mpcd_pi = 3.141592653589793;
    const unsigned int nbins = 25;
    const double dphi = mpcd_pi/static_cast<double>(nbins); // [0, pi)
    const double dtheta = 2.0*mpcd_pi/static_cast<double>(nbins); // [0, 2pi)
    std::vector<unsigned int> fphi(nbins, 0), ftheta(nbins, 0);

    hoomd::detail::Saru rng(7, 7, 91);
    mpcd::detail::SpherePointGenerator<double> gen;

    const unsigned int N = 500000;
    for (unsigned int i = 0; i < N; ++i)
        {
        double3 v;
        gen(rng, v);

        // check norm of the point and make sure it lies on the unit sphere
        const double r = slow::sqrt(v.x * v.x + v.y*v.y + v.z*v.z);
        CHECK_CLOSE(r, 1.0, tol_small);

        // z = r cos(phi)
        const double phi = std::acos(v.z / r);
        const unsigned int phi_bin = static_cast<unsigned int>(phi/dphi);
        UP_ASSERT(phi_bin < nbins);
        fphi[phi_bin] += 1;

        // bin theta
        double theta = std::atan2(v.y, v.x);
        if (theta < 0.0) theta += 2.0*mpcd_pi;
        const unsigned int theta_bin = static_cast<unsigned int>(theta/dtheta);
        UP_ASSERT(theta_bin < nbins);
        ftheta[theta_bin] += 1;
        }

    for (unsigned int i = 0; i < nbins; ++i)
        {
        const double ftheta_i = static_cast<double>(ftheta[i]) / (dtheta * N);
        const double fphi_i = static_cast<double>(fphi[i]) / (dphi * N);
        CHECK_CLOSE(ftheta_i, 1.0/(2.0*mpcd_pi), 0.05);
        CHECK_CLOSE(fphi_i, 0.5*sin(dphi*(0.5+i)), 0.05);
        }

    }

//! Check the moments of a distribution
/*!
 * \param gen Distribution generator
 * \param N Number of samples to draw
 * \param ref_mean Mean of the distribution
 * \param ref_var Variance of the distribution
 * \param ref_tol Error tolerance
 */
template<class GeneratorType>
void check_moments(GeneratorType& gen,
                   const unsigned int N,
                   const double ref_mean,
                   const double ref_var,
                   const double ref_tol)
    {
    hoomd::detail::Saru rng(7, 7, 91);

    // compute moments of the distribution
    double mean(0), var(0);
    for (unsigned int i=0; i < N; ++i)
        {
        const auto rn = gen(rng);
        mean += rn;
        var += rn*rn;
        }
    mean /= N;
    var = var/N - mean*mean;

    // check mean using close or small, depending on how close it is to zero
    if (std::abs(ref_mean) > tol_small)
        CHECK_CLOSE(mean, ref_mean, ref_tol);
    else
        CHECK_SMALL(mean, ref_tol);

    if (std::abs(ref_var) > tol_small)
        CHECK_CLOSE(var, ref_var, ref_tol);
    else
        CHECK_SMALL(var, ref_tol);
    }

//! Test case for NormalGenerator -- double, no cache
UP_TEST( normal_double_nocache_test )
    {
    mpcd::detail::NormalGenerator<double, false> gen;
    check_moments(gen, 5000000, 0.0, 1.0, 0.01);
    }
//! Test case for NormalGenerator -- double, cache
UP_TEST( normal_double_cache_test )
    {
    mpcd::detail::NormalGenerator<double, true> gen;
    check_moments(gen, 5000000, 0.0, 1.0, 0.01);
    }
//! Test case for NormalGenerator -- float, no cache
UP_TEST( normal_float_nocache_test )
    {
    mpcd::detail::NormalGenerator<float, false> gen;
    check_moments(gen, 5000000, 0.0, 1.0, 0.01);
    }
//! Test case for NormalGenerator -- float, no cache
UP_TEST( normal_float_cache_test )
    {
    mpcd::detail::NormalGenerator<float, true> gen;
    check_moments(gen, 5000000, 0.0, 1.0, 0.01);
    }

//! Test case for GammaGenerator -- double
UP_TEST( gamma_double_test )
    {
    mpcd::detail::GammaGenerator<double> gen(2.5, 2.);
    check_moments(gen, 5000000, 2.5*2, 2.5*2*2, 0.01);
    }
//! Test case for GammaGenerator -- float
UP_TEST( gamma_float_test )
    {
    mpcd::detail::GammaGenerator<float> gen(2.5, 2.);
    check_moments(gen, 5000000, 2.5*2, 2.5*2*2, 0.01);
    }

//! Test case for Saru::normal, float
UP_TEST( saru_normal_float_test )
    {
    struct generator
        {
        float operator() (hoomd::detail::Saru& rng)
            {
            return rng.normal(2.0f, 0.5f);
            }
        };
    generator gen;
    check_moments(gen, 5000000, 0.5, 4.0, 0.01);
    }

//! Test case for Saru::normal, double
UP_TEST( saru_normal_double_test )
    {
    struct generator
        {
        float operator() (hoomd::detail::Saru& rng)
            {
            return rng.normal(2.0, 0.5);
            }
        };
    generator gen;
    check_moments(gen, 5000000, 0.5, 4.0, 0.01);
    }

UP_TEST( r123_u01_range_test_float )
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    float smallest = r123::u01<float>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, (float)2.710505431213761e-20);
    float largest = r123::u01<float>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0f);
    }

UP_TEST( r123_u01_range_test_double )
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    double smallest = r123::u01<double>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, 2.710505431213761e-20);
    double largest = r123::u01<double>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0);
    }
