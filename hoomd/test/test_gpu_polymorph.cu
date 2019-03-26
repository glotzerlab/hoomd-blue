// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "test_gpu_polymorph.cuh"
#include "hoomd/GPUPolymorph.cuh"

__global__ void test_operator_kernel(int* result, const ArithmeticOperator* op, unsigned int N)
    {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    result[idx] = op->call(idx);
    }

void test_operator(int* result, const ArithmeticOperator* op, unsigned int N)
    {
    const unsigned int block_size = 32;
    const unsigned int num_blocks = (N + block_size - 1)/block_size;
    test_operator_kernel<<<num_blocks,block_size>>>(result, op, N);
    }

template AdditionOperator* hoomd::gpu::device_new(int);
template MultiplicationOperator* hoomd::gpu::device_new(int);
template void hoomd::gpu::device_delete(ArithmeticOperator*);
