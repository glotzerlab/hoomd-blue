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
        HOSTDEVICE virtual ~ArithmeticOperator() {}
        HOSTDEVICE virtual int call(int b) const = 0;
    };

class AdditionOperator : public ArithmeticOperator
    {
    public:
        HOSTDEVICE AdditionOperator(int a) : a_(a) {}

        HOSTDEVICE virtual int call(int b) const
            {
            return a_+b;
            }

    private:
        int a_;
    };

class MultiplicationOperator : public ArithmeticOperator
    {
    public:
        HOSTDEVICE MultiplicationOperator(int a) : a_(a) {}

        HOSTDEVICE virtual int call(int b) const
            {
            return a_*b;
            }

    private:
        int a_;
    };

void test_operator(int* result, const ArithmeticOperator* op, unsigned int N);

#undef HOSTDEVICE

#endif // HOOMD_TEST_TEST_GPU_POLYMORPH_CUH_
