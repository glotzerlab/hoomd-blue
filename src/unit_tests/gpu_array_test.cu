/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "gpu_array_test.cuh"

// $Id$
// $URL$
// Maintainer: joaander

/*! \file gpu_array_test.cu
    \brief GPU kernels for gpu_array_test.cc
    \ingroup unit_tests
*/

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post All \a num elements in d_data are incremented by 1
*/
__global__ void gpu_add_one_kernel(int *d_data, unsigned int num)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num)
        d_data[idx] = d_data[idx] + 1;
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_add_one is just a driver for gpu_add_one_kernel()
*/
extern "C" cudaError_t gpu_add_one(int *d_data, unsigned int num)
    {
    unsigned int block_size = 256;
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    gpu_add_one_kernel<<<grid, threads>>>(d_data, num);
    
    cudaThreadSynchronize();
    return cudaGetLastError();
    }


/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    \post Element \a i in \a d_data is set to \a i * \a i
*/
__global__ void gpu_fill_test_pattern_kernel(int *d_data, unsigned int num)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num)
        d_data[idx] = idx*idx;
    }

/*! \param d_data Device pointer to the array where the data is held
    \param num Number of elements in the array

    gpu_fill_test_pattern is just a driver for gpu_fill_test_pattern_kernel()
*/
extern "C" cudaError_t gpu_fill_test_pattern(int *d_data, unsigned int num)
    {
    unsigned int block_size = 256;
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)num / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    gpu_fill_test_pattern_kernel<<<grid, threads>>>(d_data, num);
    
    cudaThreadSynchronize();
    return cudaGetLastError();
    }
