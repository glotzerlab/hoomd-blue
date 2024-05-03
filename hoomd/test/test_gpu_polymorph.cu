// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/GPUPolymorph.cuh"
#include "test_gpu_polymorph.cuh"

__global__ void test_operator_kernel(int* result, const ArithmeticOperator* op, unsigned int N)
    {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    result[idx] = op->call(idx);
    }

void test_operator(int* result, const ArithmeticOperator* op, unsigned int N)
    {
    const unsigned int block_size = 32;
    const unsigned int num_blocks = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL((test_operator_kernel),
                       dim3(num_blocks),
                       dim3(block_size),
                       0,
                       0,
                       result,
                       op,
                       N);
    }

template AdditionOperator* hoomd::gpu::device_new(int);
template MultiplicationOperator* hoomd::gpu::device_new(int);
template void hoomd::gpu::device_delete(ArithmeticOperator*);
