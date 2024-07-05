// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepRATTLELangevinGPU.cuh"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
using namespace hoomd;

#include <assert.h>

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
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

__global__ void gpu_rattle_langevin_angular_step_two_kernel(const Scalar4* d_pos,
                                                            Scalar4* d_orientation,
                                                            Scalar4* d_angmom,
                                                            const Scalar3* d_inertia,
                                                            Scalar4* d_net_torque,
                                                            const unsigned int* d_group_members,
                                                            const Scalar3* d_gamma_r,
                                                            const unsigned int* d_tag,
                                                            size_t n_types,
                                                            unsigned int group_size,
                                                            uint64_t timestep,
                                                            uint16_t seed,
                                                            Scalar T,
                                                            Scalar tolerance,
                                                            bool noiseless_r,
                                                            Scalar deltaT,
                                                            unsigned int D,
                                                            Scalar scale)
    {
    HIP_DYNAMIC_SHARED(char, s_data)
    Scalar3* s_gammas_r = (Scalar3*)s_data;

    // read in the gamma_r, stored in s_gammas_r[0: n_type] (Pythonic convention)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas_r[cur_offset + threadIdx.x] = d_gamma_r[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        unsigned int ptag = d_tag[idx];

        // torque update with rotational drag and noise
        unsigned int type_r = __scalar_as_int(d_pos[idx].w);
        Scalar3 gamma_r = s_gammas_r[type_r];

        if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
            {
            quat<Scalar> q(d_orientation[idx]);
            quat<Scalar> p(d_angmom[idx]);
            vec3<Scalar> t(d_net_torque[idx]);
            vec3<Scalar> I(d_inertia[idx]);

            vec3<Scalar> s = (Scalar(1. / 2.) * conj(q) * p).v;

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

            // first calculate in the body frame random and damping torque imposed by the dynamics
            vec3<Scalar> bf_torque;
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
        bool x_zero = (I.x == 0);
        bool y_zero = (I.y == 0);
        bool z_zero = (I.z == 0);

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
    \param rattle_langevin_args Collected arguments for gpu_rattle_langevin_step_two_kernel() and
   gpu_rattle_langevin_angular_step_two() \param deltaT timestep \param D dimensionality of the
   system

    This is just a driver for gpu_rattle_langevin_angular_step_two_kernel(), see it for details.

*/
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
                                     Scalar scale)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid((group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    const auto shared_bytes = max((sizeof(Scalar3) * rattle_langevin_args.n_types),
                                  (rattle_langevin_args.block_size * sizeof(Scalar)));

    if (shared_bytes > rattle_langevin_args.devprop.sharedMemPerBlock)
        {
        throw std::runtime_error("Langevin gamma parameters exceed the available shared "
                                 "memory per block.");
        }

    // run the kernel
    hipLaunchKernelGGL(gpu_rattle_langevin_angular_step_two_kernel,
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
                       rattle_langevin_args.n_types,
                       group_size,
                       rattle_langevin_args.timestep,
                       rattle_langevin_args.seed,
                       rattle_langevin_args.T,
                       rattle_langevin_args.tolerance,
                       rattle_langevin_args.noiseless_r,
                       deltaT,
                       D,
                       scale);

    return hipSuccess;
    }

//! Kernel function for reducing a partial sum to a full sum (one value)
/*! \param d_sum Placeholder for the sum
    \param d_partial_sum Array containing the partial sum
    \param num_blocks Number of blocks to execute
*/
__global__ void gpu_rattle_bdtally_reduce_partial_sum_kernel(Scalar* d_sum,
                                                             Scalar* d_partial_sum,
                                                             unsigned int num_blocks)
    {
    Scalar sum = Scalar(0.0);
    extern __shared__ char s_data[];
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
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
