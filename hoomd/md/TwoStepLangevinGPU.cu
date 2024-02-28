// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepLangevinGPU.cuh"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
using namespace hoomd;

#include <assert.h>

/*! \file TwoStepLangevinGPU.cu
    \brief Defines GPU kernel code for Langevin integration on the GPU. Used by TwoStepLangevinGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Takes the second half-step forward in the Langevin integration on a group of particles with
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
__global__ void gpu_langevin_step_two_kernel(const Scalar4* d_pos,
                                             Scalar4* d_vel,
                                             Scalar3* d_accel,
                                             const unsigned int* d_tag,
                                             unsigned int* d_group_members,
                                             unsigned int group_size,
                                             Scalar4* d_net_force,
                                             Scalar* d_gamma,
                                             unsigned int n_types,
                                             uint64_t timestep,
                                             uint16_t seed,
                                             Scalar T,
                                             bool noiseless_t,
                                             Scalar deltaT,
                                             unsigned int D,
                                             bool tally,
                                             Scalar* d_partial_sum_bdenergy,
                                             bool enable_shared_cache)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* s_gammas = (Scalar*)s_data;

    if (enable_shared_cache)
        {
        // read in the gammas (1 dimensional array)
        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }

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
        if (enable_shared_cache)
            {
            gamma = s_gammas[typ];
            }
        else
            {
            gamma = d_gamma[typ];
            }

        Scalar coeff = sqrtf(Scalar(6.0) * gamma * T / deltaT);
        Scalar3 bd_force = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        if (noiseless_t)
            coeff = Scalar(0.0);

        // Initialize the Random Number Generator and generate the 3 random numbers
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevin, timestep, seed),
                            hoomd::Counter(ptag));
        UniformDistribution<Scalar> uniform(-1, 1);

        Scalar randomx = uniform(rng);
        Scalar randomy = uniform(rng);
        Scalar randomz = uniform(rng);

        bd_force.x = randomx * coeff - gamma * vel.x;
        bd_force.y = randomy * coeff - gamma * vel.y;
        if (D > 2)
            bd_force.z = randomz * coeff - gamma * vel.z;

        // read in the net force and calculate the acceleration MEM TRANSFER: 16 bytes
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
        // MEM TRANSFER: 4 bytes   FLOPS: 3
        Scalar mass = vel.w;
        Scalar minv = Scalar(1.0) / mass;
        accel.x = (accel.x + bd_force.x) * minv;
        accel.y = (accel.y + bd_force.y) * minv;
        accel.z = (accel.z + bd_force.z) * minv;

        // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
        // update the velocity (FLOPS: 6)
        vel.x += (Scalar(1.0) / Scalar(2.0)) * accel.x * deltaT;
        vel.y += (Scalar(1.0) / Scalar(2.0)) * accel.y * deltaT;
        vel.z += (Scalar(1.0) / Scalar(2.0)) * accel.z * deltaT;

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

//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the partial sum
    \param num_blocks Number of blocks to execute
*/
__global__ void
gpu_bdtally_reduce_partial_sum_kernel(Scalar* d_sum, Scalar* d_partial_sum, unsigned int num_blocks)
    {
    Scalar sum = Scalar(0.0);
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar* bdtally_sdata = (Scalar*)&s_data[0];

    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            bdtally_sdata[threadIdx.x] = d_partial_sum[start + threadIdx.x];
        else
            bdtally_sdata[threadIdx.x] = Scalar(0.0);
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

        // everybody sums up sum2K
        sum += bdtally_sdata[0];
        }

    if (threadIdx.x == 0)
        *d_sum = sum;
    }

//! NO_SQUISH angular part of the second half step
/*!
    \param d_pos array of particle positions (4th dimension is particle type)
    \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param d_gamma_r List of per-type gamma_rs (rotational drag coeff.)
    \param d_tag array of particle tags
    \param group_size Number of members in the group
    \param timestep Current timestep of the simulation
    \param seed User chosen random number seed
    \param T Temperature set point
    \param d_noiseless_r If set true, there will be no rotational noise (random torque)
    \param deltaT integration time step size
    \param D dimensionality of the system
*/

__global__ void gpu_langevin_angular_step_two_kernel(const Scalar4* d_pos,
                                                     Scalar4* d_orientation,
                                                     Scalar4* d_angmom,
                                                     const Scalar3* d_inertia,
                                                     Scalar4* d_net_torque,
                                                     const unsigned int* d_group_members,
                                                     const Scalar3* d_gamma_r,
                                                     const unsigned int* d_tag,
                                                     unsigned int n_types,
                                                     unsigned int group_size,
                                                     uint64_t timestep,
                                                     uint16_t seed,
                                                     Scalar T,
                                                     bool noiseless_r,
                                                     Scalar deltaT,
                                                     unsigned int D,
                                                     Scalar scale,
                                                     bool enable_shared_cache)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar3* s_gammas_r = (Scalar3*)s_data;

    if (enable_shared_cache)
        {
        // read in the gamma_r, stored in s_gammas_r[0: n_type] (Pythonic convention)
        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas_r[cur_offset + threadIdx.x] = d_gamma_r[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        unsigned int ptag = d_tag[idx];

        // torque update with rotational drag and noise
        unsigned int type_r = __scalar_as_int(d_pos[idx].w);

        Scalar3 gamma_r;
        if (enable_shared_cache)
            {
            gamma_r = s_gammas_r[type_r];
            }
        else
            {
            gamma_r = d_gamma_r[type_r];
            }

        if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
            {
            quat<Scalar> q(d_orientation[idx]);
            quat<Scalar> p(d_angmom[idx]);
            vec3<Scalar> t(d_net_torque[idx]);
            vec3<Scalar> I(d_inertia[idx]);

            vec3<Scalar> s;
            s = (Scalar(1. / 2.) * conj(q) * p).v;

            // first calculate in the body frame random and damping torque imposed by the dynamics
            vec3<Scalar> bf_torque;

            // original Gaussian random torque
            // for future reference: if gamma_r is different for xyz, then we need to generate 3
            // sigma_r
            Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(2.0) * gamma_r.x * T / deltaT),
                                           fast::sqrt(Scalar(2.0) * gamma_r.y * T / deltaT),
                                           fast::sqrt(Scalar(2.0) * gamma_r.z * T / deltaT));
            if (noiseless_r)
                sigma_r = make_scalar3(0, 0, 0);

            RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevinAngular, timestep, seed),
                                hoomd::Counter(ptag));
            Scalar rand_x = NormalDistribution<Scalar>(sigma_r.x)(rng);
            Scalar rand_y = NormalDistribution<Scalar>(sigma_r.y)(rng);
            Scalar rand_z = NormalDistribution<Scalar>(sigma_r.z)(rng);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x == 0);
            y_zero = (I.y == 0);
            z_zero = (I.z == 0);

            bf_torque.x = rand_x - gamma_r.x * (s.x / I.x);
            bf_torque.y = rand_y - gamma_r.y * (s.y / I.y);
            bf_torque.z = rand_z - gamma_r.z * (s.z / I.z);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero)
                bf_torque.x = 0;
            if (y_zero)
                bf_torque.y = 0;
            if (z_zero)
                bf_torque.z = 0;

            // change to lab frame and update the net torque
            bf_torque = rotate(q, bf_torque);
            d_net_torque[idx].x += bf_torque.x;
            d_net_torque[idx].y += bf_torque.y;
            d_net_torque[idx].z += bf_torque.z;

            // with the wishful mind that compiler may use conditional move to avoid branching
            if (D < 3)
                d_net_torque[idx].x = 0;
            if (D < 3)
                d_net_torque[idx].y = 0;
            }

        //////////////////////////////
        // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
        quat<Scalar> q(d_orientation[idx]);
        quat<Scalar> p(d_angmom[idx]);
        vec3<Scalar> t(d_net_torque[idx]);
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = Scalar(0.0);
        if (y_zero)
            t.y = Scalar(0.0);
        if (z_zero)
            t.z = Scalar(0.0);

        // rescale
        p = p * scale;

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT * q * t;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_pos array of particle positions (4th dimension is particle type)
    \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param d_gamma_r List of per-type gamma_rs (rotational drag coeff.)
    \param d_tag array of particle tags
    \param group_size Number of members in the group
    \param langevin_args Collected arguments for gpu_langevin_step_two_kernel() and
   gpu_langevin_angular_step_two() \param deltaT timestep \param D dimensionality of the system

    This is just a driver for gpu_langevin_angular_step_two_kernel(), see it for details.

*/
hipError_t gpu_langevin_angular_step_two(const Scalar4* d_pos,
                                         Scalar4* d_orientation,
                                         Scalar4* d_angmom,
                                         const Scalar3* d_inertia,
                                         Scalar4* d_net_torque,
                                         const unsigned int* d_group_members,
                                         const Scalar3* d_gamma_r,
                                         const unsigned int* d_tag,
                                         unsigned int group_size,
                                         const langevin_step_two_args& langevin_args,
                                         Scalar deltaT,
                                         unsigned int D,
                                         Scalar scale)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    auto shared_bytes = max((sizeof(Scalar3) * langevin_args.n_types),
                            (langevin_args.block_size * sizeof(Scalar)));

    bool enable_shared_cache = true;

    if (shared_bytes > langevin_args.devprop.sharedMemPerBlock)
        {
        enable_shared_cache = false;
        shared_bytes = 0;
        }

    // run the kernel
    hipLaunchKernelGGL(gpu_langevin_angular_step_two_kernel,
                       grid,
                       threads,
                       shared_bytes,
                       0,
                       d_pos,
                       d_orientation,
                       d_angmom,
                       d_inertia,
                       d_net_torque,
                       d_group_members,
                       d_gamma_r,
                       d_tag,
                       langevin_args.n_types,
                       group_size,
                       langevin_args.timestep,
                       langevin_args.seed,
                       langevin_args.T,
                       langevin_args.noiseless_r,
                       deltaT,
                       D,
                       scale,
                       enable_shared_cache);

    return hipSuccess;
    }

/*! \param d_pos array of particle positions and types
    \param d_vel array of particle positions and masses
    \param d_accel array of particle accelerations
    \param d_tag array of particle tags
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param langevin_args Collected arguments for gpu_langevin_step_two_kernel() and
   gpu_langevin_angular_step_two() \param deltaT Amount of real time to step forward in one time
   step \param D Dimensionality of the system

    This is just a driver for gpu_langevin_step_two_kernel(), see it for details.
*/
hipError_t gpu_langevin_step_two(const Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 Scalar3* d_accel,
                                 const unsigned int* d_tag,
                                 unsigned int* d_group_members,
                                 unsigned int group_size,
                                 Scalar4* d_net_force,
                                 const langevin_step_two_args& langevin_args,
                                 Scalar deltaT,
                                 unsigned int D)
    {
    // setup the grid to run the kernel
    dim3 grid(langevin_args.num_blocks, 1, 1);
    dim3 grid1(1, 1, 1);
    dim3 threads(langevin_args.block_size, 1, 1);
    dim3 threads1(256, 1, 1);

    auto shared_bytes = max((sizeof(Scalar) * langevin_args.n_types),
                            (langevin_args.block_size * sizeof(Scalar)));

    bool enable_shared_cache = true;

    if (shared_bytes > langevin_args.devprop.sharedMemPerBlock)
        {
        enable_shared_cache = false;
        shared_bytes = 0;
        }

    // run the kernel
    hipLaunchKernelGGL((gpu_langevin_step_two_kernel),
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
                       langevin_args.d_gamma,
                       langevin_args.n_types,
                       langevin_args.timestep,
                       langevin_args.seed,
                       langevin_args.T,
                       langevin_args.noiseless_t,
                       deltaT,
                       D,
                       langevin_args.tally,
                       langevin_args.d_partial_sum_bdenergy,
                       enable_shared_cache);

    // run the summation kernel
    if (langevin_args.tally)
        hipLaunchKernelGGL((gpu_bdtally_reduce_partial_sum_kernel),
                           dim3(grid1),
                           dim3(threads1),
                           langevin_args.block_size * sizeof(Scalar),
                           0,
                           &langevin_args.d_sum_bdenergy[0],
                           langevin_args.d_partial_sum_bdenergy,
                           langevin_args.num_blocks);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
