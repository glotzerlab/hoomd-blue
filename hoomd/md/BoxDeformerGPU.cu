// Copyright (c) 2009-2026 The Regents of the University of Michigan.
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
//! Wrap particles after box deformation
__global__ void gpu_boxdeformer_wrap_kernel(unsigned int N,
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

cudaError_t gpu_boxdeformer_wrap(const unsigned int N,
                                 Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 int3* d_image,
                                 const BoxDim& new_box,
                                 unsigned int block_size)
    {
    unsigned int max_block_size;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)gpu_boxdeformer_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    gpu_boxdeformer_wrap_kernel<<<(N / run_block_size) + 1, run_block_size>>>(N,
                                                                              d_pos,
                                                                              d_vel,
                                                                              d_image,
                                                                              new_box);

    return cudaSuccess;
    }

    } // end namespace kernel
    } // namespace md
    } // end namespace hoomd
