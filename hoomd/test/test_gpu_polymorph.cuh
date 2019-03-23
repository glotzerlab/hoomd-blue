// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_
#define HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

class ArithmeticOperator
    {
    public:
        HOSTDEVICE ArithmeticOperator(int a) : a_(a) {}
        HOSTDEVICE virtual int call(int b) const = 0;

    protected:
        int a_;
    };

class AdditionOperator : public ArithmeticOperator
    {
    public:
        HOSTDEVICE AdditionOperator(int a) : ArithmeticOperator(a) {}

        HOSTDEVICE virtual int call(int b) const
            {
            return a_+b;
            }
    };

class MultiplicationOperator : public ArithmeticOperator
    {
    public:
        HOSTDEVICE MultiplicationOperator(int a) : ArithmeticOperator(a) {}

        HOSTDEVICE virtual int call(int b) const
            {
            return a_*b;
            }
    };

void test_operator(int* result, const ArithmeticOperator* op, unsigned int N);

#undef HOSTDEVICE

#endif // HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_
