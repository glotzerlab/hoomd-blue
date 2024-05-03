// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

#include "hoomd/hpmc/IntegratorHPMCMono.h"
#include "hoomd/hpmc/Moves.h"

#include <iostream>

#include <memory>
#include <pybind11/pybind11.h>

using namespace hoomd;
using namespace std;
using namespace hoomd::hpmc;
using namespace hoomd::hpmc::detail;

struct ShapeDummy
    {
    vec3<Scalar> pos;
    quat<Scalar> orientation;
    };

UP_TEST(rand_rotate_3d)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));

    quat<Scalar> a(1, vec3<Scalar>(0, 0, 0));
    for (int i = 0; i < 10000; i++)
        {
        // move the shape
        quat<Scalar> prev = a;
        move_rotate<3>(a, rng, 1.0);
        quat<Scalar> delta(prev.s - a.s, prev.v - a.v);

        // check that all coordinates moved
        // yes, it is possible that one of the random numbers is zero - if that is the case we can
        // pick a different seed so that we do not sample that case
        UP_ASSERT(fabs(delta.s) > 0);
        UP_ASSERT(fabs(delta.v.x) > 0);
        UP_ASSERT(fabs(delta.v.y) > 0);
        UP_ASSERT(fabs(delta.v.z) > 0);

        // check that it is a valid rotation
        MY_CHECK_CLOSE(norm2(a), 1, tol);
        }
    }

UP_TEST(rand_rotate_2d)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));

    Scalar a = .1;

    quat<Scalar> o(1, vec3<Scalar>(0, 0, 0));
    for (int i = 0; i < 10000; i++)
        {
        // move the shape
        quat<Scalar> prev = o;
        move_rotate<2>(o, rng, a);
        quat<Scalar> delta(prev.s - o.s, prev.v - o.v);

        // check that the angle coordinate moved and that the 0 components stayed 0
        // yes, it is possible that one of the random numbers is zero - if that is the case we can
        // pick a different seed so that we do not sample that case
        UP_ASSERT(fabs(delta.s) > 0);
        MY_CHECK_SMALL(o.v.x, tol_small);
        MY_CHECK_SMALL(o.v.y, tol_small);
        UP_ASSERT(fabs(delta.v.z) > 0);

        // check that it is a valid rotation
        MY_CHECK_CLOSE(norm2(o), 1, tol);

        // check that the angle of the rotation is not too big
        UP_ASSERT((acos(prev.s) * 2.0 - acos(o.s) * 2.0) <= a);
        }
    }

UP_TEST(rand_translate_3d)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));
    Scalar d = 0.1;
    // test randomly generated quaternions for unit norm

    vec3<Scalar> a(0, 0, 0);
    for (int i = 0; i < 10000; i++)
        {
        // move the shape
        vec3<Scalar> prev = a;
        move_translate(a, rng, d, 3);
        vec3<Scalar> delta = prev - a;

        // check that all coordinates moved
        // yes, it is possible that one of the random numbers is zero - if that is the case we can
        // pick a different seed so that we do not sample that case
        UP_ASSERT(fabs(delta.x) > 0);
        UP_ASSERT(fabs(delta.y) > 0);
        UP_ASSERT(fabs(delta.z) > 0);

        // check that the move distance is appropriate
        UP_ASSERT(sqrt(dot(delta, delta)) <= d);
        }
    }

UP_TEST(rand_translate_2d)
    {
    hoomd::RandomGenerator rng(hoomd::Seed(0, 1, 2), hoomd::Counter(4, 5, 6));
    Scalar d = 0.1;
    // test randomly generated quaternions for unit norm

    vec3<Scalar> a(0, 0, 0);
    for (int i = 0; i < 100; i++)
        {
        // move the shape
        vec3<Scalar> prev = a;
        move_translate(a, rng, d, 2);
        vec3<Scalar> delta = prev - a;

        // check that all coordinates moved
        // yes, it is possible that one of the random numbers is zero - if that is the case we can
        // pick a different seed so that we do not sample that case
        UP_ASSERT(fabs(delta.x) > 0);
        UP_ASSERT(fabs(delta.y) > 0);
        UP_ASSERT(delta.z == 0);

        // check that the move distance is appropriate
        UP_ASSERT(sqrt(dot(delta, delta)) <= d);
        }
    }

void test_update_order(const unsigned int max)
    {
    // do a simple check on the update order, just make sure that the first index is evenly
    // distributed between 0 and N-1
    const unsigned int nsamples = 1000000;
    unsigned int counts[2];

    for (unsigned int i = 0; i < 2; i++)
        counts[i] = 0;

    UpdateOrder o(max);
    for (unsigned int i = 0; i < nsamples; i++)
        {
        o.shuffle(i, 10);
        if (o[0] == 0)
            {
            counts[0]++;
            }
        else if (o[0] == max - 1)
            {
            counts[1]++;
            }
        else
            {
            cout << "invalid count: " << o[0] << endl;
            UP_ASSERT(false);
            }
        }

    MY_CHECK_CLOSE(double(counts[0]) / double(nsamples), 0.5, 0.5);
    MY_CHECK_CLOSE(double(counts[1]) / double(nsamples), 0.5, 0.5);
    }

UP_TEST(update_order_test)
    {
    for (unsigned int max = 2; max < 10; max++)
        {
        // std::cout << "Testing max=" << max << std::endl;
        test_update_order(max);
        }
    }
