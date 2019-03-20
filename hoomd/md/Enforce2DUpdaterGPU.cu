// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "Enforce2DUpdaterGPU.cuh"

#include <assert.h>

#include <stdio.h>

/*! \file Enforce2DUpdaterGPU.cu
    \brief Defines GPU kernel code for constraining systems to a 2D plane on
    the GPU. Used by Enforce2DUpdaterGPU.
*/

//! Constrains particles to the xy plane on the GPU
/*! \param N number of particles in system
    \param d_vel Particle velocities to constrain to xy plane
    \param d_accel Particle accelerations to constrain to xy plane
*/
extern "C" __global__
void gpu_enforce2d_kernel(const unsigned int N,
                          Scalar4 *d_vel,
                          Scalar3 *d_accel)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 vel = d_vel[idx];
        Scalar3 accel = d_accel[idx];

        // zero the z-velocity and z-acceleration(FLOPS: ?)
        vel.z = Scalar(0.0);
        accel.z = Scalar(0.0);

        // write out the results (MEM_TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        d_accel[idx] = accel;
        }
    }

/*! \param N number of particles in system
    \param d_vel Particle velocities to constrain to xy plane
    \param d_accel Particle accelerations to constrain to xy plane
*/
cudaError_t gpu_enforce2d(const unsigned int N,
                          Scalar4 *d_vel,
                          Scalar3 *d_accel)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (N/block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_enforce2d_kernel<<< grid, threads >>>(N, d_vel, d_accel);

    return cudaSuccess;
    }
