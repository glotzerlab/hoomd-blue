// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ExampleUpdater.cuh"

/*! \file ExampleUpdater.cu
    \brief CUDA kernels for ExampleUpdater
*/

namespace hoomd
    {
namespace kernel
    {
// First, the kernel code for zeroing the velocities on the GPU
//! Kernel that zeroes velocities on the GPU
/*! \param d_vel Velocity-mass array from the ParticleData
    \param N Number of particles

    This kernel executes one thread per particle and zeros the velocity of each. It can be run with
   any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
__global__ void gpu_zero_velocities_kernel(Scalar4* d_vel, unsigned int N)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // vel.w is the mass, don't want to modify that
        Scalar4 vel = d_vel[idx];
        vel.x = vel.y = vel.z = 0.0f;
        d_vel[idx] = vel;
        }
    }

/*! \param d_vel Velocity-mass array from the ParticleData
    \param N Number of particles
    This is just a driver for gpu_zero_velocities_kernel(), see it for the details
*/
hipError_t gpu_zero_velocities(Scalar4* d_vel, unsigned int N)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((int)ceil((double)N / (double)block_size), 1, 1);
    if (grid.x == 0)
        {
        grid.x = 1;
        }
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL(gpu_zero_velocities_kernel, dim3(grid), dim3(threads), 0, 0, d_vel, N);

    // this method always succeeds. If you had a cuda* call in this driver, you could return its
    // error code if not hipSuccess
    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace hoomd
