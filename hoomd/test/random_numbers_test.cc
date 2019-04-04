// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/RandomNumbers.h"
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
    hoomd::detail::SpherePointGenerator<double> gen;

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
    hoomd::detail::RandomGenerator rng(7, 7, 91);

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
    // std::cout << "mean: " << ref_mean << " " << mean << std::endl;
    if (std::abs(ref_mean) > tol_small)
        CHECK_CLOSE(mean, ref_mean, ref_tol);
    else
        CHECK_SMALL(mean, ref_tol);

    // std::cout << "variance: " << ref_var << " " << var << std::endl;
    if (std::abs(ref_var) > tol_small)
        CHECK_CLOSE(var, ref_var, ref_tol);
    else
        CHECK_SMALL(var, ref_tol);
    }

//! Check the range of a distribution
/*! \param gen Distribution generator
    \param N Number of samples to draw
    \param a Minimum of range (inclusive)
    \param b Maximum  of range (inclusive)
 */
template<class ValueType, class GeneratorType>
void check_range(GeneratorType& gen,
                 const unsigned int N,
                 const ValueType a,
                 const ValueType b)
    {
    hoomd::detail::RandomGenerator rng(1, 2, 3);

    // check that every value generated is in the proper range
    for (unsigned int i=0; i < N; ++i)
        {
        const auto rn = gen(rng);
        if (! (rn >= a))
            {
            std::cout << "Out of range: " << rn <<std::endl;
            }
        UP_ASSERT(rn >= a);
        if (! (rn <= b))
            {
            std::cout << "Out of range: " << rn <<std::endl;
            }
        UP_ASSERT(rn <= b);
        }
    }

//! Test case for NormalGenerator
UP_TEST( normal_double_test )
    {
    hoomd::detail::NormalDistribution<double> gen(2.0, 1.5);
    check_moments(gen, 5000000, 1.5, 4.0, 0.01);
    }
//! Test case for NormalGenerator
UP_TEST( normal_default_double_test )
    {
    hoomd::detail::NormalDistribution<double> gen;
    check_moments(gen, 5000000, 0.0, 1.0, 0.01);
    }
//! Test case for NormalGenerator -- float, no cache
UP_TEST( normal_float_test )
    {
    hoomd::detail::NormalDistribution<double> gen(2.0f, 1.5f);
    check_moments(gen, 5000000, 1.5, 4.0, 0.01);
    }

//! Test case for GammaDistribution -- double
UP_TEST( gamma_double_test )
    {
    hoomd::detail::GammaDistribution<double> gen(2.5, 2.);
    check_moments(gen, 5000000, 2.5*2, 2.5*2*2, 0.01);
    }
//! Test case for GammaDistribution -- float
UP_TEST( gamma_float_test )
    {
    hoomd::detail::GammaDistribution<float> gen(2.5, 2.);
    check_moments(gen, 5000000, 2.5*2, 2.5*2*2, 0.01);
    }

UP_TEST( r123_u01_range_test_float )
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    float smallest = r123::u01<float>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, (float)2.710505431213761e-20);
    float largest = r123::u01<float>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0f);
    }

UP_TEST( canonical_float_moment )
    {
    struct gen
        {
        float operator()(hoomd::detail::RandomGenerator& rng)
            {
            return hoomd::detail::generate_canonical<float>(rng);
            }
        };
    gen canonical;
    check_moments(canonical, 5000000, 0.5, 1.0/12.0, 0.01);
    check_range(canonical, 5000000, 2.710505431213761e-20f, 1.0f);
    }

UP_TEST( r123_u01_range_test_double )
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    double smallest = r123::u01<double>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, 2.710505431213761e-20);
    double largest = r123::u01<double>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0);
    }

UP_TEST( canonical_double_moment )
    {
    struct gen
        {
        double operator()(hoomd::detail::RandomGenerator& rng)
            {
            return hoomd::detail::generate_canonical<double>(rng);
            }
        };
    gen canonical;
    check_moments(canonical, 5000000, 0.5, 1.0/12.0, 0.01);
    check_range(canonical, 5000000, 2.710505431213761e-20, 1.0);
    }

//! Test case for UniformDistribution -- double
UP_TEST( uniform_double_test )
    {
    hoomd::detail::UniformDistribution<double> gen(1, 3);
    check_moments(gen, 5000000, 2.0, 1.0/3.0, 0.01);
    check_range(gen, 5000000, 1.0f, 3.0f);
    }
//! Test case for UniformDistribution -- float
UP_TEST( uniform_float_test )
    {
    hoomd::detail::UniformDistribution<float> gen(-4, 0);
    check_moments(gen, 5000000, -2.0, 1.0/12.0*4.0*4.0, 0.01);
    check_range(gen, 5000000, -4.0f, 0.0f);
    }

//! Test case for UniformIntDistribution
UP_TEST( uniform_int_test_1000 )
    {
    hoomd::detail::UniformIntDistribution gen(1000);
    check_range(gen, 5000000, uint32_t(0), uint32_t(1000));
    check_moments(gen, 5000000, 500, 1.0/12.0*1000*1000, 0.01);
    }

//! Test case for UniformIntDistribution
UP_TEST( uniform_int_test_256 )
    {
    hoomd::detail::UniformIntDistribution gen(256);
    check_range(gen, 5000000, uint32_t(0), uint32_t(256));
    check_moments(gen, 5000000, 128.0, 1.0/12.0*256*256, 0.01);
    }
