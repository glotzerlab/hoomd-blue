// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_
#define HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_

#include <hip/hip_runtime.h>

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

class ArithmeticOperator
    {
    public:
    HOSTDEVICE virtual ~ArithmeticOperator() { }
    HOSTDEVICE virtual int call(int b) const = 0;
    };

class AdditionOperator : public ArithmeticOperator
    {
    public:
    HOSTDEVICE AdditionOperator(int a) : a_(a) { }

    HOSTDEVICE virtual int call(int b) const
        {
        return a_ + b;
        }

    private:
    int a_;
    };

class MultiplicationOperator : public ArithmeticOperator
    {
    public:
    HOSTDEVICE MultiplicationOperator(int a) : a_(a) { }

    HOSTDEVICE virtual int call(int b) const
        {
        return a_ * b;
        }

    private:
    int a_;
    };

void test_operator(int* result, const ArithmeticOperator* op, unsigned int N);

#undef HOSTDEVICE

#endif // HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_
