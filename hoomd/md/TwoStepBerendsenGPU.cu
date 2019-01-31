// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepBerendsenGPU.cuh"

#include <assert.h>

/*! \file TwoStepBerendsenGPU.cu
    \brief CUDA kernels for BerendsenGPU
*/

// First, the kernel code for the Berendsen thermostat
//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for applying periodic boundary conditions
    \param lambda Intermediate variable computed on the host and used in integrating the velocity
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the thermostat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_one_kernel(Scalar4 *d_pos,
                                   Scalar4 *d_vel,
                                   const Scalar3 *d_accel,
                                   int3 *d_image,
                                   unsigned int *d_group_members,
                                   const unsigned int group_size,
                                   const BoxDim box,
                                   const Scalar lambda,
                                   const Scalar deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read the particle position
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        // read the particle velocity and acceleration
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar3 accel = d_accel[idx];

        // integrate velocity and position forward in time
        vel = lambda * (vel + accel * deltaT / Scalar(2.0));
        pos += vel * deltaT;

        // read in the image flags
        int3 image = d_image[idx];

        // apply the periodic boundary conditions
        box.wrap(pos, image);

        // write the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

//! Kernel that applies the first step of a Berendsen integration to a group of particles
/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Current net force on the particles
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the thermostat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_berendsen_step_two_kernel(Scalar4 *d_vel,
                                   Scalar3 *d_accel,
                                   unsigned int *d_group_members,
                                   const unsigned int group_size,
                                   const Scalar4 *d_net_force,
                                   const Scalar deltaT)
    {
    // determine the particle index for this thread
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // read in the velocity
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);
        Scalar mass = velmass.w;

        // read in the net force and calculate the acceleration
        Scalar4 net_force_energy = d_net_force[idx];
        Scalar3 net_force = make_scalar3(net_force_energy.x, net_force_energy.y, net_force_energy.z);
        Scalar3 accel = net_force / mass;

        // integrate the velocity
        vel += accel * deltaT / Scalar(2.0);

        // write out the velocity and acceleration
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_accel[idx] = accel;
        }
    }

cudaError_t gpu_berendsen_step_one(Scalar4 *d_pos,
                                   Scalar4 *d_vel,
                                   const Scalar3 *d_accel,
                                   int3 *d_image,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   const BoxDim& box,
                                   unsigned int block_size,
                                   Scalar lambda,
                                   Scalar deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_berendsen_step_one_kernel<<< grid, threads, block_size * sizeof(Scalar) >>>(d_pos,
                                                                                   d_vel,
                                                                                   d_accel,
                                                                                   d_image,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   box,
                                                                                   lambda,
                                                                                   deltaT);

    return cudaSuccess;
    }

cudaError_t gpu_berendsen_step_two(Scalar4 *d_vel,
                                   Scalar3 *d_accel,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   Scalar4 *d_net_force,
                                   unsigned int block_size,
                                   Scalar deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_berendsen_step_two_kernel<<< grid, threads, block_size * sizeof(Scalar) >>>(d_vel,
                                                                                   d_accel,
                                                                                   d_group_members,
                                                                                   group_size,
                                                                                   d_net_force,
                                                                                   deltaT);

    return cudaSuccess;
    }
