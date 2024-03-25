// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <hip/hip_runtime.h>
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file TwoStepRATTLELangevinGPU.cuh
    \brief Declares GPU kernel code for RATTLELangevin dynamics on the GPU. Used by
   TwoStepRATTLELangevinGPU.
*/

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <assert.h>
#include <type_traits>

#ifndef __TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__
#define __TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Temporary holder struct to limit the number of arguments passed to
//! gpu_rattle_langevin_step_two()
struct rattle_langevin_step_two_args
    {
    rattle_langevin_step_two_args(Scalar* _d_gamma,
                                  size_t _n_types,
                                  Scalar _T,
                                  Scalar _tolerance,
                                  uint64_t _timestep,
                                  uint16_t _seed,
                                  Scalar* _d_sum_bdenergy,
                                  Scalar* _d_partial_sum_bdenergy,
                                  unsigned int _block_size,
                                  unsigned int _num_blocks,
                                  bool _noiseless_t,
                                  bool _noiseless_r,
                                  bool _tally,
                                  const hipDeviceProp_t& _devprop)
        : d_gamma(_d_gamma), n_types(_n_types), T(_T), tolerance(_tolerance), timestep(_timestep),
          seed(_seed), d_sum_bdenergy(_d_sum_bdenergy),
          d_partial_sum_bdenergy(_d_partial_sum_bdenergy), block_size(_block_size),
          num_blocks(_num_blocks), noiseless_t(_noiseless_t), noiseless_r(_noiseless_r),
          tally(_tally), devprop(_devprop)
        {
        }

    Scalar* d_gamma; //!< Device array listing per-type gammas
    size_t n_types;  //!< Number of types in \a d_gamma
    Scalar T;        //!< Current temperature
    Scalar tolerance;
    uint64_t timestep;              //!< Current timestep
    uint16_t seed;                  //!< User chosen random number seed
    Scalar* d_sum_bdenergy;         //!< Energy transfer sum from bd thermal reservoir
    Scalar* d_partial_sum_bdenergy; //!< Array used for summation
    unsigned int block_size;        //!<  Block size
    unsigned int num_blocks;        //!<  Number of blocks
    bool noiseless_t; //!<  If set true, there will be no translational noise (random force)
    bool noiseless_r; //!<  If set true, there will be no rotational noise (random torque)
    bool tally;       //!< Set to true is bd thermal reservoir energy ally is to be performed
    const hipDeviceProp_t& devprop; //!< Device properties.
    };

hipError_t
gpu_rattle_langevin_angular_step_two(const Scalar4* d_pos,
                                     Scalar4* d_orientation,
                                     Scalar4* d_angmom,
                                     const Scalar3* d_inertia,
                                     Scalar4* d_net_torque,
                                     const unsigned int* d_group_members,
                                     const Scalar3* d_gamma_r,
                                     const unsigned int* d_tag,
                                     unsigned int group_size,
                                     const rattle_langevin_step_two_args& rattle_langevin_args,
                                     Scalar deltaT,
                                     unsigned int D,
                                     Scalar scale);

__global__ void gpu_rattle_bdtally_reduce_partial_sum_kernel(Scalar* d_sum,
                                                             Scalar* d_partial_sum,
                                                             unsigned int num_blocks);

template<class Manifold>
hipError_t gpu_rattle_langevin_step_two(const Scalar4* d_pos,
                                        Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        const unsigned int* d_tag,
                                        unsigned int* d_group_members,
                                        unsigned int group_size,
                                        Scalar4* d_net_force,
                                        const rattle_langevin_step_two_args& rattle_langevin_args,
                                        Manifold manifold,
                                        Scalar deltaT,
                                        unsigned int D);

#ifdef __HIPCC__

/*! \file TwoStepRATTLELangevinGPU.cu
    \brief Defines GPU kernel code for RATTLELangevin integration on the GPU. Used by
   TwoStepRATTLELangevinGPU.
*/

//! Takes the second half-step forward in the RATTLELangevin integration on a group of particles
//! with
/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param d_gamma List of per-type gammas
    \param n_types Number of particle types in the simulation
    \param timestep Current timestep of the simulation
    \param seed User chosen random number seed
    \param T Temperature set point
    \param deltaT Amount of real time to step forward in one time step
    \param D Dimensionality of the system
    \param tally Boolean indicating whether energy tally is performed or not
    \param d_partial_sum_bdenergy Placeholder for the partial sum

    This kernel is implemented in a very similar manner to gpu_nve_step_two_kernel(), see it for
   design details.

    This kernel will tally the energy transfer from the bd thermal reservoir and the particle system

    This kernel must be launched with enough dynamic shared memory per block to read in d_gamma
*/

template<class Manifold>
__global__ void gpu_rattle_langevin_step_two_kernel(const Scalar4* d_pos,
                                                    Scalar4* d_vel,
                                                    Scalar3* d_accel,
                                                    const unsigned int* d_tag,
                                                    unsigned int* d_group_members,
                                                    unsigned int group_size,
                                                    Scalar4* d_net_force,
                                                    Scalar* d_gamma,
                                                    size_t n_types,
                                                    uint64_t timestep,
                                                    uint16_t seed,
                                                    Scalar T,
                                                    Scalar tolerance,
                                                    bool noiseless_t,
                                                    Manifold manifold,
                                                    Scalar deltaT,
                                                    unsigned int D,
                                                    bool tally,
                                                    Scalar* d_partial_sum_bdenergy)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* s_gammas = (Scalar*)s_data;

    // read in the gammas (1 dimensional array)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar bd_energy_transfer = 0;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // ******** first, calculate the additional BD force
        // read the current particle velocity (MEM TRANSFER: 16 bytes)
        Scalar4 vel = d_vel[idx];
        // read in the tag of our particle.
        // (MEM TRANSFER: 4 bytes)
        unsigned int ptag = d_tag[idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // read in the type of our particle. A texture read of only the fourth part of the
        // position Scalar4 (where type is stored) is used.
        unsigned int typ = __scalar_as_int(d_pos[idx].w);
        gamma = s_gammas[typ];

        // read in the net force and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);

        Scalar3 pos = make_scalar3(d_pos[idx].x, d_pos[idx].y, d_pos[idx].z);

        // Initialize the Random Number Generator and generate the 3 random numbers
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevin, timestep, seed),
                            hoomd::Counter(ptag));

        Scalar3 normal = manifold.derivative(pos);
        Scalar ndotn = dot(normal, normal);

        Scalar randomx, randomy, randomz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(-1, 1);

            randomx = uniform(rng);
            randomy = uniform(rng);
            randomz = uniform(rng);

            coeff = sqrtf(Scalar(6.0) * gamma * T / deltaT);
            Scalar3 bd_force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
            if (noiseless_t)
                coeff = Scalar(0.0);

            Scalar proj_x = normal.x / fast::sqrt(ndotn);
            Scalar proj_y = normal.y / fast::sqrt(ndotn);
            Scalar proj_z = normal.z / fast::sqrt(ndotn);

            Scalar proj_r = randomx * proj_x + randomy * proj_y + randomz * proj_z;
            randomx = randomx - proj_r * proj_x;
            randomy = randomy - proj_r * proj_y;
            randomz = randomz - proj_r * proj_z;
            }
        else
            {
            randomx = 0;
            randomy = 0;
            randomz = 0;
            coeff = 0;
            }

        Scalar3 bd_force;

        bd_force.x = randomx * coeff - gamma * vel.x;
        bd_force.y = randomy * coeff - gamma * vel.y;
        bd_force.z = randomz * coeff - gamma * vel.z;

        // MEM TRANSFER: 4 bytes   FLOPS: 3
        Scalar mass = vel.w;
        Scalar minv = Scalar(1.0) / mass;
        accel.x = (accel.x + bd_force.x) * minv;
        accel.y = (accel.y + bd_force.y) * minv;
        accel.z = (accel.z + bd_force.z) * minv;

        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
        // update the velocity (FLOPS: 6)

        Scalar3 next_vel;
        next_vel.x = vel.x + Scalar(1.0 / 2.0) * deltaT * accel.x;
        next_vel.y = vel.y + Scalar(1.0 / 2.0) * deltaT * accel.y;
        next_vel.z = vel.z + Scalar(1.0 / 2.0) * deltaT * accel.z;

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 residual;
        Scalar resid;
        Scalar3 vel_dot;

        const unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            vel_dot.x = accel.x - mu * minv * normal.x;
            vel_dot.y = accel.y - mu * minv * normal.y;
            vel_dot.z = accel.z - mu * minv * normal.z;

            residual.x = vel.x - next_vel.x + Scalar(1.0 / 2.0) * deltaT * vel_dot.x;
            residual.y = vel.y - next_vel.y + Scalar(1.0 / 2.0) * deltaT * vel_dot.y;
            residual.z = vel.z - next_vel.z + Scalar(1.0 / 2.0) * deltaT * vel_dot.z;
            resid = dot(normal, next_vel) * minv;

            Scalar ndotr = dot(normal, residual);
            Scalar beta = (mass * resid + ndotr) / ndotn;
            next_vel.x = next_vel.x - normal.x * beta + residual.x;
            next_vel.y = next_vel.y - normal.y * beta + residual.y;
            next_vel.z = next_vel.z - normal.z * beta + residual.z;
            mu = mu - mass * beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid * mass > tolerance && iteration < maxiteration);

        vel.x += (Scalar(1.0) / Scalar(2.0)) * (accel.x - mu * minv * normal.x) * deltaT;
        vel.y += (Scalar(1.0) / Scalar(2.0)) * (accel.y - mu * minv * normal.y) * deltaT;
        vel.z += (Scalar(1.0) / Scalar(2.0)) * (accel.z - mu * minv * normal.z) * deltaT;

        // tally the energy transfer from the bd thermal reservoir to the particles (FLOPS: 6)
        bd_energy_transfer = bd_force.x * vel.x + bd_force.y * vel.y + bd_force.z * vel.z;

        // write out data (MEM TRANSFER: 32 bytes)
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }

    Scalar* bdtally_sdata = (Scalar*)&s_data[0];
    if (tally)
        {
        // don't overwrite values in the s_gammas array with bd_energy transfer
        __syncthreads();
        bdtally_sdata[threadIdx.x] = bd_energy_transfer;
        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                bdtally_sdata[threadIdx.x] += bdtally_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        // write out our partial sum
        if (threadIdx.x == 0)
            {
            d_partial_sum_bdenergy[blockIdx.x] = bdtally_sdata[0];
            }
        }
    }

/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param rattle_langevin_args Collected arguments for gpu_rattle_langevin_step_two_kernel() and
   gpu_rattle_langevin_angular_step_two() \param deltaT Amount of real time to step forward in one
   time step \param D Dimensionality of the system

    This is just a driver for gpu_rattle_langevin_step_two_kernel(), see it for details.
*/
template<class Manifold>
hipError_t gpu_rattle_langevin_step_two(const Scalar4* d_pos,
                                        Scalar4* d_vel,
                                        Scalar3* d_accel,
                                        const unsigned int* d_tag,
                                        unsigned int* d_group_members,
                                        unsigned int group_size,
                                        Scalar4* d_net_force,
                                        const rattle_langevin_step_two_args& rattle_langevin_args,
                                        Manifold manifold,
                                        Scalar deltaT,
                                        unsigned int D)
    {
    // setup the grid to run the kernel
    dim3 grid(rattle_langevin_args.num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(rattle_langevin_args.block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    size_t shared_bytes = max((unsigned int)(sizeof(Scalar) * rattle_langevin_args.n_types),
                              (unsigned int)(rattle_langevin_args.block_size * sizeof(Scalar)));

    if (shared_bytes > rattle_langevin_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Langevin gamma parameters exceed the available shared "
                                 "memory per block.");
        }

    // run the kernel
    hipLaunchKernelGGL((gpu_rattle_langevin_step_two_kernel<Manifold>),
                       grid,
                       threads,
                       shared_bytes,
                       0,
                       d_pos,
                       d_vel,
                       d_accel,
                       d_tag,
                       d_group_members,
                       group_size,
                       d_net_force,
                       rattle_langevin_args.d_gamma,
                       rattle_langevin_args.n_types,
                       rattle_langevin_args.timestep,
                       rattle_langevin_args.seed,
                       rattle_langevin_args.T,
                       rattle_langevin_args.tolerance,
                       rattle_langevin_args.noiseless_t,
                       manifold,
                       deltaT,
                       D,
                       rattle_langevin_args.tally,
                       rattle_langevin_args.d_partial_sum_bdenergy);

    // run the summation kernel
    if (rattle_langevin_args.tally)
        hipLaunchKernelGGL((gpu_rattle_bdtally_reduce_partial_sum_kernel),
                           dim3(grid1),
                           dim3(threads1),
                           rattle_langevin_args.block_size * sizeof(Scalar),
                           0,
                           &rattle_langevin_args.d_sum_bdenergy[0],
                           rattle_langevin_args.d_partial_sum_bdenergy,
                           rattle_langevin_args.num_blocks);

    return hipSuccess;
    }

#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__
