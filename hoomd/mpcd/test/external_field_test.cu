// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#include "external_field_test.cuh"
#include "hoomd/GPUPolymorph.cuh"

namespace gpu
{
namespace kernel
{
__global__ void test_external_field(Scalar3* out, const mpcd::ExternalField* field, const Scalar3* pos, const unsigned int N)
    {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = field->evaluate(pos[idx]);
    }
} // end namespace kernel

cudaError_t test_external_field(Scalar3* out, const mpcd::ExternalField* field, const Scalar3* pos, const unsigned int N)
    {
    const unsigned int block_size = 32;
    const unsigned int num_blocks = (N + block_size - 1)/block_size;
    kernel::test_external_field<<<num_blocks,block_size>>>(out, field, pos, N);
    return cudaSuccess;
    }
} // end namespace gpu
