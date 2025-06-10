// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"

#include "hoomd/test/upp11_config.h"
#include <vector>
using namespace hoomd;

HOOMD_UP_MAIN()

template<class T> void test_int_pow_exact()
    {
    // test powf(T, int) functions for powers of 2, which should be exact
    T base = 2.0;
    std::vector<std::pair<int, T>> expected {
        {-3, T(1.0 / 8.0)},
        {-2, T(1.0 / 4.0)},
        {-1, T(1.0 / 2.0)},
        {0, T(1.0)},
        {1, T(2.0)},
        {2, T(4.0)},
        {3, T(8.0)},
    };
    for (auto&& test_v : expected)
        {
        MY_ASSERT_EQUAL(::fast::pow(base, test_v.first), test_v.second);
        }
    }

template<class T> void test_int_pow_close()
    {
    // test powf(T, int) functions and compare them against std::powf
    std::vector<T> bases {T(-3.234), T(-0.343), T(0.6545), T(2.5844)};
    for (int exponent = -5; exponent < 5; exponent++)
        {
        for (auto&& base : bases)
            MY_CHECK_CLOSE(::fast::pow(base, exponent), pow(base, T(exponent)), tol_small);
        }
    }

UP_TEST(integer_power)
    {
    test_int_pow_exact<double>();
    test_int_pow_exact<float>();
    test_int_pow_close<double>();
    test_int_pow_close<float>();
    }
