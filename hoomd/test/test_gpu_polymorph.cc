// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file test_gpu_polymorph.cc
 * \brief Tests for GPUPolymorph wrapper.
 */

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GPUArray.h"

#include "hoomd/GPUPolymorph.h"
#include "test_gpu_polymorph.cuh"

#include "upp11_config.h"
HOOMD_UP_MAIN();

void test_gpu_polymorph(const ExecutionConfiguration::executionMode mode)
    {
    auto exec_conf = std::make_shared<ExecutionConfiguration>(mode);

    // default initialization is empty
    hoomd::GPUPolymorph<ArithmeticOperator> add_op(exec_conf);
    auto times_op = std::make_shared<hoomd::GPUPolymorph<ArithmeticOperator>>(exec_conf);
    UP_ASSERT(add_op.get(access_location::host) == nullptr);
    UP_ASSERT(times_op->get(access_location::host) == nullptr);

    // resetting should allocate these members
    add_op.reset<AdditionOperator>(7);
    const int a = 7;
    times_op->reset<MultiplicationOperator>(a);

    // check host polymorphism
    UP_ASSERT(add_op.get(access_location::host) != nullptr);
    UP_ASSERT(times_op->get(access_location::host) != nullptr);
    UP_ASSERT_EQUAL(add_op.get(access_location::host)->call(3), 10);
    UP_ASSERT_EQUAL(times_op->get(access_location::host)->call(3), 21);

    // check device polymorphism
    #ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        UP_ASSERT(add_op.get(access_location::device) != nullptr);
        UP_ASSERT(times_op->get(access_location::device) != nullptr);

        GPUArray<int> result(2, exec_conf);
        // addition operator
            {
            ArrayHandle<int> d_result(result, access_location::device, access_mode::overwrite);
            test_operator(d_result.data, add_op.get(access_location::device), 2);
            }
            {
            ArrayHandle<int> h_result(result, access_location::host, access_mode::read);
            UP_ASSERT_EQUAL(h_result.data[0], 7);
            UP_ASSERT_EQUAL(h_result.data[1], 8);
            }
        // multiplication operator
            {
            ArrayHandle<int> d_result(result, access_location::device, access_mode::overwrite);
            test_operator(d_result.data, times_op->get(access_location::device), 2);
            }
            {
            ArrayHandle<int> h_result(result, access_location::host, access_mode::read);
            UP_ASSERT_EQUAL(h_result.data[0], 0);
            UP_ASSERT_EQUAL(h_result.data[1], 7);
            }
        }
    else
        {
        UP_ASSERT(add_op.get(access_location::device) == nullptr);
        UP_ASSERT(times_op->get(access_location::device) == nullptr);
        }
    #endif // ENABLE_CUDA
    }

//! Test polymorphism on CPU
UP_TEST( test_gpu_polymorph_cpu )
    {
    test_gpu_polymorph(ExecutionConfiguration::CPU);
    }

#ifdef ENABLE_CUDA
//! Test polymorphism on GPU
UP_TEST( test_gpu_polymorph_gpu )
    {
    test_gpu_polymorph(ExecutionConfiguration::GPU);
    }
#endif // ENABLE_CUDA
