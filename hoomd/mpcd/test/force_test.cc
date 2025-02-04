// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <vector>

#include "hoomd/mpcd/BlockForce.h"
#include "hoomd/mpcd/ConstantForce.h"
#include "hoomd/mpcd/NoForce.h"
#include "hoomd/mpcd/SineForce.h"

#include "hoomd/test/upp11_config.h"

HOOMD_UP_MAIN();
using namespace hoomd;

template<class Force>
void test_force(std::shared_ptr<Force> force,
                const std::vector<Scalar3>& ref_pos,
                const std::vector<Scalar3>& ref_force)
    {
    UP_ASSERT_EQUAL(ref_pos.size(), ref_force.size());
    for (unsigned int i = 0; i < ref_pos.size(); ++i)
        {
        const Scalar3 r = ref_pos[i];
        const Scalar3 F = force->evaluate(r);
        const Scalar3 F_ref = ref_force[i];
        UP_ASSERT_CLOSE(F.x, F_ref.x, tol_small);
        UP_ASSERT_CLOSE(F.y, F_ref.y, tol_small);
        UP_ASSERT_CLOSE(F.z, F_ref.z, tol_small);
        }
    }

//! Test block force
UP_TEST(block_force)
    {
    auto force = std::make_shared<mpcd::BlockForce>(2, 6, 0.4);
    std::vector<Scalar3> ref_pos = {make_scalar3(0, 2.7, 0),
                                    make_scalar3(1, 2.9, 2),
                                    make_scalar3(2, 3.3, 1),
                                    make_scalar3(-1, -2, 0),
                                    make_scalar3(4, -3, 5),
                                    make_scalar3(4, -4, 5)};
    std::vector<Scalar3> ref_force = {make_scalar3(0, 0, 0),
                                      make_scalar3(2, 0, 0),
                                      make_scalar3(0, 0, 0),
                                      make_scalar3(0, 0, 0),
                                      make_scalar3(-2, 0, 0),
                                      make_scalar3(0, 0, 0)};
    test_force(force, ref_pos, ref_force);
    }

//! Test constant force
UP_TEST(constant_force)
    {
    auto force = std::make_shared<mpcd::ConstantForce>(make_scalar3(6, 7, 8));
    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 3), make_scalar3(-1, 0, -2)};
    std::vector<Scalar3> ref_force = {make_scalar3(6, 7, 8), make_scalar3(6, 7, 8)};
    test_force(force, ref_pos, ref_force);
    }

//! Test no force (zeros)
UP_TEST(no_force)
    {
    auto force = std::make_shared<mpcd::NoForce>();
    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 3), make_scalar3(-1, 0, -2)};
    std::vector<Scalar3> ref_force = {make_scalar3(0, 0, 0), make_scalar3(0, 0, 0)};
    test_force(force, ref_pos, ref_force);
    }

//! Test sine force
UP_TEST(sine_force)
    {
    auto force = std::make_shared<mpcd::SineForce>(Scalar(2.0), Scalar(M_PI));
    std::vector<Scalar3> ref_pos = {make_scalar3(1, 0.5, 2), make_scalar3(-1, -1. / 6., 0)};
    std::vector<Scalar3> ref_force = {make_scalar3(2.0, 0, 0), make_scalar3(-1., 0, 0)};
    test_force(force, ref_pos, ref_force);
    }
