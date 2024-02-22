// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ClockSource.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <iomanip>
#include <iostream>
#include <vector>

#include "hoomd/test/upp11_config.h"
using namespace hoomd;

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
UP_TEST(sphere_point_test)
    {
    // initialize the histograms
    const double mpcd_pi = 3.141592653589793;
    const unsigned int nbins = 25;
    const double dphi = mpcd_pi / static_cast<double>(nbins);         // [0, pi)
    const double dtheta = 2.0 * mpcd_pi / static_cast<double>(nbins); // [0, 2pi)
    std::vector<unsigned int> fphi(nbins, 0), ftheta(nbins, 0);

    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));
    hoomd::SpherePointGenerator<double> gen;

    const unsigned int N = 500000;
    for (unsigned int i = 0; i < N; ++i)
        {
        double3 v;
        gen(rng, v);

        // check norm of the point and make sure it lies on the unit sphere
        const double r = slow::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        CHECK_CLOSE(r, 1.0, tol_small);

        // z = r cos(phi)
        const double phi = std::acos(v.z / r);
        const unsigned int phi_bin = static_cast<unsigned int>(phi / dphi);
        UP_ASSERT(phi_bin < nbins);
        fphi[phi_bin] += 1;

        // bin theta
        double theta = std::atan2(v.y, v.x);
        if (theta < 0.0)
            theta += 2.0 * mpcd_pi;
        const unsigned int theta_bin = static_cast<unsigned int>(theta / dtheta);
        UP_ASSERT(theta_bin < nbins);
        ftheta[theta_bin] += 1;
        }

    for (unsigned int i = 0; i < nbins; ++i)
        {
        const double ftheta_i = static_cast<double>(ftheta[i]) / (dtheta * N);
        const double fphi_i = static_cast<double>(fphi[i]) / (dphi * N);
        CHECK_CLOSE(ftheta_i, 1.0 / (2.0 * mpcd_pi), 0.05);
        CHECK_CLOSE(fphi_i, 0.5 * sin(dphi * (0.5 + i)), 0.05);
        }
    }

//! Kahan summation
class KahanSum
    {
    public:
    KahanSum(double _s)
        {
        sum = _s;
        }

    void operator+=(double x)
        {
        // from https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        double y = x - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
        }

    operator double()
        {
        return sum;
        }

    private:
    double sum;
    double c = 0.0;
    };

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
                   const double ref_skew,
                   const double ref_exkurtosis,
                   const double ref_tol,
                   bool test_kurtosis = true)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));

    // compute moments of the distribution
    // use Kahan summation to prevent errors when summing over many samples
    KahanSum sample_x(0), sample_x2(0), sample_x3(0), sample_x4(0);
    std::vector<double> v(N);

    double n = double(N);

    for (unsigned int i = 0; i < N; ++i)
        {
        const auto rn = gen(rng);
        sample_x += rn;
        v[i] = rn;
        }

    double mean = sample_x / n;

    // TODO: block sums to avoid round off error
    for (unsigned int i = 0; i < N; ++i)
        {
        // use two pass method to compute moments
        // this is more numerically stable: See
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance and unbiased
        double x = v[i] - mean;

        sample_x2 += x * x;
        sample_x3 += x * x * x;
        sample_x4 += x * x * x * x;
        }
    double var = sample_x2 / (n - 1);
    // sample skewness: https://en.wikipedia.org/wiki/Skewness
    double skew = (1.0 / n) * sample_x3 / (sqrt(var) * sqrt(var) * sqrt(var));
    // sample excess kurtosis: https://en.wikipedia.org/wiki/Kurtosis
    double exkurtosis
        = (n + 1) * n * (n - 1) * sample_x4 / ((n - 2) * (n - 3) * sample_x2 * sample_x2)
          - 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));

    // check mean using close or small, depending on how close it is to zero
    if (std::abs(ref_mean) > tol_small)
        CHECK_CLOSE(mean, ref_mean, ref_tol);
    else
        CHECK_SMALL(mean, ref_tol);

    if (std::abs(ref_var) > tol_small)
        CHECK_CLOSE(var, ref_var, ref_tol);
    else
        CHECK_SMALL(var, ref_tol);

    if (std::abs(ref_skew) > tol_small)
        CHECK_CLOSE(skew, ref_skew, ref_tol);
    else
        CHECK_SMALL(skew, ref_tol);

    // skip kurtosis checks for distributions where it is finicky
    if (test_kurtosis)
        {
        if (std::abs(ref_exkurtosis) > tol_small)
            CHECK_CLOSE(exkurtosis, ref_exkurtosis, ref_tol);
        else
            CHECK_SMALL(exkurtosis, ref_tol);
        }
    }

//! Check the range of a distribution
/*! \param gen Distribution generator
    \param N Number of samples to draw
    \param a Minimum of range (inclusive)
    \param b Maximum  of range (inclusive)
 */
template<class ValueType, class GeneratorType>
void check_range(GeneratorType& gen, const unsigned int N, const ValueType a, const ValueType b)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));

    // check that every value generated is in the proper range
    for (unsigned int i = 0; i < N; ++i)
        {
        const auto rn = gen(rng);
        if (!(rn >= a))
            {
            std::cout << "Out of range: " << rn << std::endl;
            }
        UP_ASSERT(rn >= a);
        if (!(rn <= b))
            {
            std::cout << "Out of range: " << rn << std::endl;
            }
        UP_ASSERT(rn <= b);
        }
    }

//! Test case for NormalDistribution
UP_TEST(normal_double_test)
    {
    double mu = 1.5, sigma = 2.0;
    double mean = mu, var = sigma * sigma, skew = 0, exkurtosis = 0.0;
    hoomd::NormalDistribution<double> gen(sigma, mu);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    }
//! Test case for NormalDistribution
UP_TEST(normal_default_double_test)
    {
    double mu = 0.0, sigma = 1.0;
    double mean = mu, var = sigma * sigma, skew = 0, exkurtosis = 0.0;
    hoomd::NormalDistribution<double> gen;
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    }
//! Test case for NormalDistribution -- float
UP_TEST(normal_float_test)
    {
    float mu = 2.0, sigma = 1.5;
    double mean = mu, var = sigma * sigma, skew = 0, exkurtosis = 0.0;
    hoomd::NormalDistribution<float> gen(sigma, mu);
    check_moments(gen, 500000, mean, var, exkurtosis, skew, 0.01);
    }

//! Test case for GammaDistribution -- double (alpha=1/2)
UP_TEST(gamma_double_half_alpha)
    {
    double alpha = 0.5, b = 2.0;
    double mean = alpha * b, var = alpha * b * b, skew = 2.0 / sqrt(alpha),
           exkurtosis = 6.0 / alpha;
    hoomd::GammaDistribution<double> gen(alpha, b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    }

//! Test case for GammaDistribution -- double (small alpha)
UP_TEST(gamma_double_small_alpha_test)
    {
    double alpha = 2.5, b = 2.0;
    double mean = alpha * b, var = alpha * b * b, skew = 2.0 / sqrt(alpha),
           exkurtosis = 6.0 / alpha;
    hoomd::GammaDistribution<double> gen(alpha, b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    }
//! Test case for GammaDistribution -- double (large N)
UP_TEST(gamma_double_large_alpha_test)
    {
    double alpha = 1000.0, b = 2.0;
    double mean = alpha * b, var = alpha * b * b, skew = 2.0 / sqrt(alpha),
           exkurtosis = 6.0 / alpha;
    hoomd::GammaDistribution<double> gen(alpha, b);

    // Test with reduced tolerance. The skew and kurtosis computations suffer from round-off
    // errors for large alpha. Can lower tolerance when check_moments implements kahan summation.
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.05, false);
    }
//! Test case for GammaDistribution -- float
UP_TEST(gamma_float_test)
    {
    float alpha = 2.5, b = 2.0;
    double mean = alpha * b, var = alpha * b * b, skew = 2.0 / sqrt(alpha),
           exkurtosis = 6.0 / alpha;
    hoomd::GammaDistribution<float> gen(alpha, b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    }

UP_TEST(r123_u01_range_test_float)
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    float smallest = r123::u01<float>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, (float)2.710505431213761e-20);
    float largest = r123::u01<float>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0f);
    }

UP_TEST(canonical_float_moment)
    {
    struct gen
        {
        float operator()(hoomd::RandomGenerator& rng)
            {
            return hoomd::detail::generate_canonical<float>(rng);
            }
        };

    float a = 2.710505431213761e-20f, b = 1.0f;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    gen canonical;
    check_moments(canonical, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(canonical, 5000000, a, b);
    }

UP_TEST(r123_u01_range_test_double)
    {
    // equality tests on floats intentional, they validate the exact range of the RNG output
    double smallest = r123::u01<double>(uint64_t(0x0000000000000000));
    UP_ASSERT_EQUAL(smallest, 2.710505431213761e-20);
    double largest = r123::u01<double>(uint64_t(0xffffffffffffffff));
    UP_ASSERT_EQUAL(largest, 1.0);
    }

UP_TEST(canonical_double_moment)
    {
    struct gen
        {
        double operator()(hoomd::RandomGenerator& rng)
            {
            return hoomd::detail::generate_canonical<double>(rng);
            }
        };

    double a = 2.710505431213761e-20, b = 1.0;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    gen canonical;
    check_moments(canonical, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(canonical, 5000000, a, b);
    }

//! Test case for UniformDistribution -- double
UP_TEST(uniform_double_test)
    {
    double a = 1, b = 3;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    hoomd::UniformDistribution<double> gen(a, b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(gen, 5000000, a, b);
    }
//! Test case for UniformDistribution -- float
UP_TEST(uniform_float_test)
    {
    float a = -4, b = 0;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    hoomd::UniformDistribution<float> gen(a, b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(gen, 5000000, a, b);
    }

//! Test case for UniformIntDistribution
UP_TEST(uniform_int_test_1000)
    {
    uint32_t a = 0, b = 1000;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    hoomd::UniformIntDistribution gen(b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(gen, 5000000, a, b);
    }

//! Test case for UniformIntDistribution
UP_TEST(uniform_int_test_256)
    {
    uint32_t a = 0, b = 256;
    double mean = (a + b) / 2.0, var = 1.0 / 12.0 * (b - a) * (b - a), skew = 0.0,
           exkurtosis = -6.0 / 5.0;

    hoomd::UniformIntDistribution gen(b);
    check_moments(gen, 5000000, mean, var, skew, exkurtosis, 0.01);
    check_range(gen, 5000000, a, b);
    }

// use a wider tolerance and skip kurtosis checks for the Poisson distribution. These measures are
// finicky for this non-continuous distribution.

//! Test case for PoissonDistribution -- double
UP_TEST(poisson_small_double_test)
    {
    double m = 10;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<double> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test case for PoissonDistribution -- double
UP_TEST(poisson_medium_double_test)
    {
    double m = 20;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<double> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test case for PoissonDistribution -- double
UP_TEST(poisson_large_double_test)
    {
    double m = 120;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<double> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test case for PoissonDistribution -- float
UP_TEST(poisson_small_float_test)
    {
    float m = 10;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<float> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test case for PoissonDistribution -- float
UP_TEST(poisson_medium_float_test)
    {
    float m = 20;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<float> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test case for PoissonDistribution -- float
UP_TEST(poisson_large_float_test)
    {
    float m = 120;
    double mean = m, var = m, skew = 1.0 / sqrt(m), exkurtosis = 1.0 / m;

    hoomd::PoissonDistribution<float> gen(m);
    check_moments(gen, 4000000, mean, var, skew, exkurtosis, 0.03, false);
    }

//! Test that Seed initializes correctly
UP_TEST(seed_fromIDStepSeed)
    {
    auto s = hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShuffle, 0xabcdef1234567890, 0x5eed);

    UP_ASSERT_EQUAL(s.getKey()[0], 0x015eed12);
    UP_ASSERT_EQUAL(s.getKey()[1], 0x34567890);
    }

//! Test that Counter initializes correctly
UP_TEST(counter)
    {
    auto a = hoomd::Counter();

    UP_ASSERT_EQUAL(a.getCounter()[0], 0);
    UP_ASSERT_EQUAL(a.getCounter()[1], 0);
    UP_ASSERT_EQUAL(a.getCounter()[2], 0);
    UP_ASSERT_EQUAL(a.getCounter()[3], 0);

    auto b = hoomd::Counter(0xabcdef12);

    UP_ASSERT_EQUAL(b.getCounter()[0], 0);
    UP_ASSERT_EQUAL(b.getCounter()[1], 0);
    UP_ASSERT_EQUAL(b.getCounter()[2], 0);
    UP_ASSERT_EQUAL(b.getCounter()[3], 0xabcdef12);

    auto c = hoomd::Counter(0x1234, 0x5678);

    UP_ASSERT_EQUAL(c.getCounter()[0], 0);
    UP_ASSERT_EQUAL(c.getCounter()[1], 0);
    UP_ASSERT_EQUAL(c.getCounter()[2], 0x5678);
    UP_ASSERT_EQUAL(c.getCounter()[3], 0x1234);

    auto d = hoomd::Counter(0xabcd, 0xef123, 0x4567);

    UP_ASSERT_EQUAL(d.getCounter()[0], 0);
    UP_ASSERT_EQUAL(d.getCounter()[1], 0x4567);
    UP_ASSERT_EQUAL(d.getCounter()[2], 0xef123);
    UP_ASSERT_EQUAL(d.getCounter()[3], 0xabcd);

    auto e = hoomd::Counter(0xabcd, 0xef123, 0x4567, 0x1234);

    UP_ASSERT_EQUAL(e.getCounter()[0], 0x12340000);
    UP_ASSERT_EQUAL(e.getCounter()[1], 0x4567);
    UP_ASSERT_EQUAL(e.getCounter()[2], 0xef123);
    UP_ASSERT_EQUAL(e.getCounter()[3], 0xabcd);
    }

UP_TEST(rng_seeding)
    {
    auto s = hoomd::Seed(hoomd::RNGIdentifier::HPMCMonoShuffle, 0xabcdef1234567890, 0x5eed);
    auto c = hoomd::Counter(0x9876, 0x5432, 0x10fe);

    auto g = hoomd::RandomGenerator(s, c);
    UP_ASSERT_EQUAL(g.getKey()[0], 0x015eed12);
    UP_ASSERT_EQUAL(g.getKey()[1], 0x34567890);

    UP_ASSERT_EQUAL(g.getCounter()[0], 0);
    UP_ASSERT_EQUAL(g.getCounter()[1], 0x10fe);
    UP_ASSERT_EQUAL(g.getCounter()[2], 0x5432);
    UP_ASSERT_EQUAL(g.getCounter()[3], 0x9876);

    g();
    UP_ASSERT_EQUAL(g.getCounter()[0], 1);
    UP_ASSERT_EQUAL(g.getCounter()[1], 0x10fe);
    UP_ASSERT_EQUAL(g.getCounter()[2], 0x5432);
    UP_ASSERT_EQUAL(g.getCounter()[3], 0x9876);

    g();
    UP_ASSERT_EQUAL(g.getCounter()[0], 2);
    UP_ASSERT_EQUAL(g.getCounter()[1], 0x10fe);
    UP_ASSERT_EQUAL(g.getCounter()[2], 0x5432);
    UP_ASSERT_EQUAL(g.getCounter()[3], 0x9876);
    }

// //! Find performance crossover
// /*! Note: this code was written for a one time use to find the empirical crossover. It requires
// that the private:
//     be commented out in PoissonDistribution.
// */
// UP_TEST( poisson_perf_test )
//     {
//     unsigned int N = 1000000;
//     double sum=0;

//     hoomd::RandomGenerator rng(7, 7, 91);

//     std::vector<double> small, large;
//     for (int mean = 1; mean < 20; mean++)
//         {
//         hoomd::PoissonDistribution<double> gen(mean);

//             {
//             ClockSource t;
//             for (int i = 0; i < N; i++)
//                 sum += gen.poissrnd_small(rng);
//             small.push_back(double(t.getTime()) / double(N));
//             }

//             {
//             ClockSource t;
//             for (int i = 0; i < N; i++)
//                 sum += gen.poissrnd_large(rng);
//             large.push_back(double(t.getTime()) / double(N));
//             }
//         }

//     for (int i = 0; i < small.size(); i++)
//         {
//         std::cout << i+1 << " "  << small[i] << " " << large[i] << std::endl;
//         }
//     }
