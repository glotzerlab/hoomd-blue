#include "ExampleUpdater.cuh"

/*! \file ExampleUpdater.cu
    \brief CUDA kernels for ExampleUpdater
*/

// First, the kernel code for zeroing the velocities on the GPU
//! Kernel that zeroes velocities on the GPU
/*! \param pdata Particle data arrays to zero the velocities of
    
    This kernel executes one thread per particle and zeros the velocity of each. It can be run with any 1D block size
    as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__ 
void gpu_zero_velocities_kernel(gpu_pdata_arrays pdata)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pdata.N)
        {
        pdata.vel[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

/*! \param pdata Particle data arrays to zero the velocities of
    This is just a driver for gpu_zero_velocities_kernel(), see it for the details
*/
cudaError_t gpu_zero_velocities(const gpu_pdata_arrays &pdata)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)pdata.N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_zero_velocities_kernel<<< grid, threads >>>(pdata);
    
    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
    }

