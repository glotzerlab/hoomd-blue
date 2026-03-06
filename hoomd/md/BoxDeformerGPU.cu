// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/BoxDeformerGPU.cu
    \brief Definition of CUDA kernels for md::BoxDeformerGPU
*/

#include "BoxDeformerGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void gpu_boxdeformer_wrap_kernel(const unsigned int N,
                                            Scalar4* d_pos,
                                            Scalar4* d_vel,
                                            int3* d_image,
                                            const BoxDim new_box)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        new_box.wrap(d_pos[idx], d_vel[idx], d_image[idx]);
        }
    }

hipError_t gpu_boxdeformer_wrap(const unsigned int N,
                                Scalar4* d_pos,
                                Scalar4* d_vel,
                                int3* d_image,
                                const BoxDim& new_box,
                                unsigned int block_size)
    {
    unsigned int max_block_size;

    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_boxdeformer_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid((N / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_boxdeformer_wrap_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       N,
                       d_pos,
                       d_vel,
                       d_image,
                       new_box);

    return hipSuccess;
    }

    } // end namespace kernel
    } // namespace md
    } // end namespace hoomd
