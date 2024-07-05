// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file external_field_test.cc
 * \brief Tests for ExternalField functors.
 */

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GPUArray.h"

#include "hoomd/GPUPolymorph.h"
#include "hoomd/mpcd/ExternalField.h"
#ifdef ENABLE_HIP
#include "external_field_test.cuh"
#endif // ENABLE_HIP

#include <vector>

#include "hoomd/test/upp11_config.h"
HOOMD_UP_MAIN();

using namespace hoomd;

void test_external_field(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                         std::shared_ptr<hoomd::GPUPolymorph<mpcd::ExternalField>> field,
                         const std::vector<Scalar3>& ref_pos,
                         const std::vector<Scalar3>& ref_force)
    {
    const unsigned int N = ref_pos.size();

    // test points
    GPUArray<Scalar3> pos(N, exec_conf);
        {
        ArrayHandle<Scalar3> h_pos(pos, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < N; ++i)
            h_pos.data[i] = ref_pos[i];
        }

    // check host evaluation
    UP_ASSERT(field->get(access_location::host) != nullptr);
        {
        ArrayHandle<Scalar3> h_pos(pos, access_location::host, access_mode::read);
        for (unsigned int i = 0; i < N; ++i)
            {
            const Scalar3 r = h_pos.data[i];
            const Scalar3 f = field->get(access_location::host)->evaluate(r);
            UP_ASSERT_CLOSE(f.x, ref_force[i].x, tol_small);
            UP_ASSERT_CLOSE(f.y, ref_force[i].y, tol_small);
            UP_ASSERT_CLOSE(f.z, ref_force[i].z, tol_small);
            }
        }

#ifdef ENABLE_HIP
    if (exec_conf->isCUDAEnabled())
        {
        UP_ASSERT(field->get(access_location::device) != nullptr);
        GPUArray<Scalar3> out(N, exec_conf);
            {
            ArrayHandle<Scalar3> d_out(out, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar3> d_pos(pos, access_location::device, access_mode::read);
            gpu::test_external_field(d_out.data,
                                     field->get(access_location::device),
                                     d_pos.data,
                                     N);
            }
            {
            ArrayHandle<Scalar3> h_out(out, access_location::host, access_mode::read);
            for (unsigned int i = 0; i < N; ++i)
                {
                const Scalar3 f = h_out.data[i];
                UP_ASSERT_CLOSE(f.x, ref_force[i].x, tol_small);
                UP_ASSERT_CLOSE(f.y, ref_force[i].y, tol_small);
                UP_ASSERT_CLOSE(f.z, ref_force[i].z, tol_small);
                }
            }
        }
    else
        {
        UP_ASSERT(field->get(access_location::device) == nullptr);
        }
#endif // ENABLE_HIP
    }

//! Test constant force on CPU
UP_TEST(constant_force_cpu)
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::CPU);

    auto field = std::make_shared<hoomd::GPUPolymorph<mpcd::ExternalField>>(exec_conf);
    field->reset<mpcd::ConstantForce>(make_scalar3(6, 7, 8));

    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 3), make_scalar3(-1, 0, -2)};
    std::vector<Scalar3> ref_force = {make_scalar3(6, 7, 8), make_scalar3(6, 7, 8)};

    test_external_field(exec_conf, field, ref_pos, ref_force);
    }

//! Test sine force on CPU
UP_TEST(sine_force_cpu)
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::CPU);

    auto field = std::make_shared<hoomd::GPUPolymorph<mpcd::ExternalField>>(exec_conf);
    field->reset<mpcd::SineForce>(Scalar(2.0), Scalar(M_PI));

    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 0.5), make_scalar3(-1, 0, -1. / 6.)};
    std::vector<Scalar3> ref_force = {make_scalar3(2.0, 0, 0), make_scalar3(-1., 0, 0)};

    test_external_field(exec_conf, field, ref_pos, ref_force);
    }

#ifdef ENABLE_HIP
//! Test constant force on GPU
UP_TEST(constant_force_gpu)
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);

    auto field = std::make_shared<hoomd::GPUPolymorph<mpcd::ExternalField>>(exec_conf);
    field->reset<mpcd::ConstantForce>(make_scalar3(6, 7, 8));

    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 3), make_scalar3(-1, 0, -2)};
    std::vector<Scalar3> ref_force = {make_scalar3(6, 7, 8), make_scalar3(6, 7, 8)};

    test_external_field(exec_conf, field, ref_pos, ref_force);
    }

//! Test sine force on GPU
UP_TEST(sine_force_gpu)
    {
    auto exec_conf = std::make_shared<const ExecutionConfiguration>(ExecutionConfiguration::GPU);

    auto field = std::make_shared<hoomd::GPUPolymorph<mpcd::ExternalField>>(exec_conf);
    field->reset<mpcd::SineForce>(Scalar(2.0), Scalar(M_PI));

    std::vector<Scalar3> ref_pos = {make_scalar3(1, 2, 0.5), make_scalar3(-1, 0, -1. / 6.)};
    std::vector<Scalar3> ref_force = {make_scalar3(2.0, 0, 0), make_scalar3(-1., 0, 0)};

    test_external_field(exec_conf, field, ref_pos, ref_force);
    }
#endif // ENABLE_HIP
