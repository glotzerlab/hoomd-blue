// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "test_gpu_array.cuh"

/*! \file gpu_array_test.cu
    \brief GPU kernels for gpu_array_test.cc
    \ingroup unit_tests
*/

namespace hoomd
    {
namespace test
    {
/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post All \a num elements in d_data are incremented by 1
*/
__global__ void gpu_add_one_kernel(int* d_data, size_t num)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
        d_data[idx] = d_data[idx] + 1;
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_add_one is just a driver for gpu_add_one_kernel()
*/
cudaError_t gpu_add_one(int* d_data, size_t num)
    {
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    unsigned int n_blocks = (unsigned int)ceil((double)num / (double)block_size);

    gpu_add_one_kernel<<<n_blocks, block_size>>>(d_data, num);

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post Element \a i in \a d_data is set to \a i * \a i
*/
__global__ void gpu_fill_test_pattern_kernel(int* d_data, size_t num)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num)
        d_data[idx] = idx * idx;
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_fill_test_pattern is just a driver for gpu_fill_test_pattern_kernel()
*/
cudaError_t gpu_fill_test_pattern(int* d_data, size_t num)
    {
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    unsigned int n_blocks = (unsigned int)ceil((double)num / (double)block_size);

    gpu_fill_test_pattern_kernel<<<n_blocks, block_size>>>(d_data, num);

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
    }

    } // end namespace test
    } // end namespace hoomd
