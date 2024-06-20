// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file TwoStepRATTLEBDGPU.cuh
    \brief Declares GPU kernel code for Brownian dynamics on the GPU. Used by TwoStepRATTLEBDGPU.
*/

#pragma once

#include "hoomd/CachedAllocator.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/VectorMath.h"

#include "hoomd/GPUPartition.cuh"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <assert.h>
#include <type_traits>

#ifndef __TWO_STEP_RATTLE_BD_GPU_CUH__
#define __TWO_STEP_RATTLE_BD_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Temporary holder struct to limit the number of arguments passed to gpu_rattle_bd_step_one()
struct rattle_bd_step_one_args
    {
    rattle_bd_step_one_args(Scalar* _d_gamma,
                            size_t _n_types,
                            Scalar _T,
                            Scalar _tolerance,
                            uint64_t _timestep,
                            uint16_t _seed,
                            const hipDeviceProp_t& _devprop)
        : d_gamma(_d_gamma), n_types(_n_types), T(_T), tolerance(_tolerance), timestep(_timestep),
          seed(_seed), devprop(_devprop)
        {
        }

    Scalar* d_gamma; //!< Device array listing per-type gammas
    size_t n_types;  //!< Number of types in \a d_gamma
    Scalar T;        //!< Current temperature
    Scalar tolerance;
    uint64_t timestep;              //!< Current timestep
    uint16_t seed;                  //!< User chosen random number seed
    const hipDeviceProp_t& devprop; //!< Device properties.
    };

template<class Manifold>
hipError_t gpu_rattle_brownian_step_one(Scalar4* d_pos,
                                        int3* d_image,
                                        Scalar4* d_vel,
                                        const BoxDim& box,
                                        const unsigned int* d_tag,
                                        const unsigned int* d_group_members,
                                        const unsigned int group_size,
                                        const Scalar4* d_net_force,
                                        const Scalar3* d_gamma_r,
                                        Scalar4* d_orientation,
                                        Scalar4* d_torque,
                                        const Scalar3* d_inertia,
                                        Scalar4* d_angmom,
                                        const rattle_bd_step_one_args& rattle_bd_args,
                                        Manifold manifold,
                                        const bool aniso,
                                        const Scalar deltaT,
                                        const unsigned int D,
                                        const bool d_noiseless_t,
                                        const bool d_noiseless_r,
                                        const GPUPartition& gpu_partition);

template<class Manifold>
hipError_t gpu_include_rattle_force_bd(const Scalar4* d_pos,
                                       Scalar4* d_net_force,
                                       Scalar* d_net_virial,
                                       const unsigned int* d_tag,
                                       const unsigned int* d_group_members,
                                       const unsigned int group_size,
                                       const rattle_bd_step_one_args& rattle_bd_args,
                                       Manifold manifold,
                                       size_t net_virial_pitch,
                                       const Scalar deltaT,
                                       const bool d_noiseless_t,
                                       const GPUPartition& gpu_partition);

#ifdef __HIPCC__

template<class Manifold>
__global__ void gpu_rattle_brownian_step_one_kernel(Scalar4* d_pos,
                                                    int3* d_image,
                                                    Scalar4* d_vel,
                                                    const BoxDim box,
                                                    const unsigned int* d_tag,
                                                    const unsigned int* d_group_members,
                                                    const unsigned int nwork,
                                                    const Scalar4* d_net_force,
                                                    const Scalar3* d_gamma_r,
                                                    Scalar4* d_orientation,
                                                    Scalar4* d_torque,
                                                    const Scalar3* d_inertia,
                                                    Scalar4* d_angmom,
                                                    const Scalar* d_gamma,
                                                    const size_t n_types,
                                                    const uint64_t timestep,
                                                    const uint16_t seed,
                                                    const Scalar T,
                                                    Manifold manifold,
                                                    const bool aniso,
                                                    const Scalar deltaT,
                                                    unsigned int D,
                                                    const bool d_noiseless_t,
                                                    const bool d_noiseless_r,
                                                    const unsigned int offset)
    {
    HIP_DYNAMIC_SHARED(char, s_data)

    Scalar3* s_gammas_r = (Scalar3*)s_data;
    Scalar* s_gammas = (Scalar*)(s_gammas_r + n_types);

    // read in the gamma (1 dimensional array), stored in s_gammas[0: n_type] (Pythonic
    // convention)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // read in the gamma_r, stored in s_gammas_r[0: n_type], which is s_gamma_r[0:n_type]

    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas_r[cur_offset + threadIdx.x] = d_gamma_r[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_idx < nwork)
        {
        const unsigned int group_idx = local_idx + offset;

        // determine the particle to work on
        unsigned int idx = d_group_members[group_idx];
        unsigned int tag = d_tag[idx];

        Scalar4 postype = d_pos[idx];
        Scalar4 vel = d_vel[idx];
        Scalar4 net_force = d_net_force[idx];
        int3 image = d_image[idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // determine gamma from type
        unsigned int typ = __scalar_as_int(postype.w);
        gamma = s_gammas[typ];

        Scalar deltaT_gamma = deltaT / gamma;

        // compute the random force
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(tag, 1));

        // compute the random force
        RandomGenerator rng_b(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                              hoomd::Counter(tag, 2));

        Scalar3 next_pos;
        next_pos.x = postype.x;
        next_pos.y = postype.y;
        next_pos.z = postype.z;
        Scalar3 normal = manifold.derivative(next_pos);

        if (d_noiseless_t)
            {
            vel.x = net_force.x / gamma;
            vel.y = net_force.y / gamma;
            vel.z = net_force.y / gamma;
            }
        else
            {
            // draw a new random velocity for particle j
            Scalar mass = vel.w;
            Scalar sigma = fast::sqrt(T / mass);
            NormalDistribution<Scalar> norm(sigma);
            vel.x = norm(rng);
            vel.y = norm(rng);
            vel.z = norm(rng);
            }

        Scalar norm_normal = fast::rsqrt(dot(normal, normal));
        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rand_norm = vel.x * normal.x + vel.y * normal.y + vel.z * normal.z;
        vel.x -= rand_norm * normal.x;
        vel.y -= rand_norm * normal.y;
        vel.z -= rand_norm * normal.z;

        Scalar rx, ry, rz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * T / deltaT_gamma);
            if (d_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        Scalar dx = (net_force.x + rx * coeff) * deltaT_gamma;
        Scalar dy = (net_force.y + ry * coeff) * deltaT_gamma;
        Scalar dz = (net_force.z + rz * coeff) * deltaT_gamma;

        postype.x += dx;
        postype.y += dy;
        postype.z += dz;
        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(postype, image);

        // write out data
        d_pos[idx] = postype;
        d_image[idx] = image;
        d_vel[idx] = vel;

        // rotational random force and orientation quaternion updates
        if (aniso)
            {
            unsigned int type_r = __scalar_as_int(d_pos[idx].w);

            // gamma_r is stored in the second half of s_gammas a.k.a s_gammas_r
            Scalar3 gamma_r = s_gammas_r[type_r];
            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                vec3<Scalar> p_vec;
                quat<Scalar> q(d_orientation[idx]);
                vec3<Scalar> t(d_torque[idx]);
                vec3<Scalar> I(d_inertia[idx]);

                // check if the shape is degenerate
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x == 0);
                y_zero = (I.y == 0);
                z_zero = (I.z == 0);

                Scalar3 sigma_r = make_scalar3(fast::sqrt(Scalar(2.0) * gamma_r.x * T / deltaT),
                                               fast::sqrt(Scalar(2.0) * gamma_r.y * T / deltaT),
                                               fast::sqrt(Scalar(2.0) * gamma_r.z * T / deltaT));
                if (d_noiseless_r)
                    sigma_r = make_scalar3(0, 0, 0);

                // original Gaussian random torque
                // Gaussian random distribution is preferred in terms of preserving the exact math
                vec3<Scalar> bf_torque;
                bf_torque.x = NormalDistribution<Scalar>(sigma_r.x)(rng);
                bf_torque.y = NormalDistribution<Scalar>(sigma_r.y)(rng);
                bf_torque.z = NormalDistribution<Scalar>(sigma_r.z)(rng);

                if (x_zero)
		    {
                    bf_torque.x = 0;
                    t.x = 0;
		    }
                if (y_zero)
		    {
                    bf_torque.y = 0;
                    t.y = 0;
		    }
                if (z_zero)
		    {
                    bf_torque.z = 0;
                    t.z = 0;
		    }

                // use the damping by gamma_r and rotate back to lab frame
                // For Future Updates: take special care when have anisotropic gamma_r
                bf_torque = rotate(q, bf_torque);
                if (D < 3)
                    {
                    bf_torque.x = 0;
                    bf_torque.y = 0;
                    t.x = 0;
                    t.y = 0;
                    }

                // do the integration for quaternion
                q += Scalar(0.5) * deltaT * ((t + bf_torque) / vec3<Scalar>(gamma_r)) * q;
                q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
                d_orientation[idx] = quat_to_scalar4(q);

                if (d_noiseless_r)
                    {
                    p_vec.x = t.x / gamma_r.x;
                    p_vec.y = t.y / gamma_r.y;
                    p_vec.z = t.z / gamma_r.z;
                    }
                else
                    {
                    // draw a new random ang_mom for particle j in body frame
                    p_vec.x = NormalDistribution<Scalar>(fast::sqrt(T * I.x))(rng);
                    p_vec.y = NormalDistribution<Scalar>(fast::sqrt(T * I.y))(rng);
                    p_vec.z = NormalDistribution<Scalar>(fast::sqrt(T * I.z))(rng);
                    }

                if (x_zero)
                    p_vec.x = 0;
                if (y_zero)
                    p_vec.y = 0;
                if (z_zero)
                    p_vec.z = 0;

                // !! Note this ang_mom isn't well-behaving in 2D,
                // !! because may have effective non-zero ang_mom in x,y

                // store ang_mom quaternion
                quat<Scalar> p = Scalar(2.0) * q * p_vec;
                d_angmom[idx] = quat_to_scalar4(p);
                }
            }
        }
    }

template<class Manifold>
hipError_t gpu_rattle_brownian_step_one(Scalar4* d_pos,
                                        int3* d_image,
                                        Scalar4* d_vel,
                                        const BoxDim& box,
                                        const unsigned int* d_tag,
                                        const unsigned int* d_group_members,
                                        const unsigned int group_size,
                                        const Scalar4* d_net_force,
                                        const Scalar3* d_gamma_r,
                                        Scalar4* d_orientation,
                                        Scalar4* d_torque,
                                        const Scalar3* d_inertia,
                                        Scalar4* d_angmom,
                                        const rattle_bd_step_one_args& rattle_bd_args,
                                        Manifold manifold,
                                        const bool aniso,
                                        const Scalar deltaT,
                                        const unsigned int D,
                                        const bool d_noiseless_t,
                                        const bool d_noiseless_r,
                                        const GPUPartition& gpu_partition)
    {
    unsigned int run_block_size = 256;

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        const auto shared_bytes
            = (sizeof(Scalar) * rattle_bd_args.n_types + sizeof(Scalar3) * rattle_bd_args.n_types);

        if (shared_bytes > rattle_bd_args.devprop.sharedMemPerBlock)
            {
            throw std::runtime_error("Brownian gamma parameters exceed the available shared "
                                     "memory per block.");
            }

        // run the kernel
        hipLaunchKernelGGL((gpu_rattle_brownian_step_one_kernel<Manifold>),
                           dim3(grid),
                           dim3(threads),
                           shared_bytes,
                           0,
                           d_pos,
                           d_image,
                           d_vel,
                           box,
                           d_tag,
                           d_group_members,
                           nwork,
                           d_net_force,
                           d_gamma_r,
                           d_orientation,
                           d_torque,
                           d_inertia,
                           d_angmom,
                           rattle_bd_args.d_gamma,
                           rattle_bd_args.n_types,
                           rattle_bd_args.timestep,
                           rattle_bd_args.seed,
                           rattle_bd_args.T,
                           manifold,
                           aniso,
                           deltaT,
                           D,
                           d_noiseless_t,
                           d_noiseless_r,
                           range.first);
        }

    return hipSuccess;
    }

template<class Manifold>
__global__ void gpu_include_rattle_force_bd_kernel(const Scalar4* d_pos,
                                                   Scalar4* d_net_force,
                                                   Scalar* d_net_virial,
                                                   const unsigned int* d_tag,
                                                   const unsigned int* d_group_members,
                                                   const unsigned int nwork,
                                                   const Scalar* d_gamma,
                                                   const size_t n_types,
                                                   const uint64_t timestep,
                                                   const uint16_t seed,
                                                   const Scalar T,
                                                   const Scalar tolerance,
                                                   Manifold manifold,
                                                   size_t net_virial_pitch,
                                                   const Scalar deltaT,
                                                   const bool d_noiseless_t,
                                                   const unsigned int offset)
    {
    HIP_DYNAMIC_SHARED(char, s_data2)

    Scalar3* s_gammas_r = (Scalar3*)s_data2;
    Scalar* s_gammas = (Scalar*)(s_gammas_r + n_types);

    // read in the gamma (1 dimensional array), stored in s_gammas[0: n_type] (Pythonic
    // convention)
    for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < n_types)
            s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
        }
    __syncthreads();

    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_idx < nwork)
        {
        const unsigned int group_idx = local_idx + offset;

        // determine the particle to work on
        unsigned int idx = d_group_members[group_idx];
        unsigned int tag = d_tag[idx];

        Scalar4 postype = d_pos[idx];
        Scalar4 net_force = d_net_force[idx];
        Scalar3 brownian_force = make_scalar3(0, 0, 0);

        Scalar virial0 = d_net_virial[0 * net_virial_pitch + idx];
        Scalar virial1 = d_net_virial[1 * net_virial_pitch + idx];
        Scalar virial2 = d_net_virial[2 * net_virial_pitch + idx];
        Scalar virial3 = d_net_virial[3 * net_virial_pitch + idx];
        Scalar virial4 = d_net_virial[4 * net_virial_pitch + idx];
        Scalar virial5 = d_net_virial[5 * net_virial_pitch + idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        // determine gamma from type
        unsigned int typ = __scalar_as_int(postype.w);
        gamma = s_gammas[typ];

        Scalar deltaT_gamma = deltaT / gamma;

        // compute the random force
        RandomGenerator rng_b(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                              hoomd::Counter(tag, 2));

        Scalar3 next_pos;
        next_pos.x = postype.x;
        next_pos.y = postype.y;
        next_pos.z = postype.z;

        Scalar3 normal = manifold.derivative(next_pos);
        Scalar norm_normal = fast::rsqrt(dot(normal, normal));
        normal.x *= norm_normal;
        normal.y *= norm_normal;
        normal.z *= norm_normal;

        Scalar rx, ry, rz, coeff;

        if (T > 0)
            {
            UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
            rx = uniform(rng_b);
            ry = uniform(rng_b);
            rz = uniform(rng_b);

            Scalar normal_r = rx * normal.x + ry * normal.y + rz * normal.z;

            rx = rx - normal_r * normal.x;
            ry = ry - normal_r * normal.y;
            rz = rz - normal_r * normal.z;

            // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform
            // -1,1 distribution it is not the dimensionality of the system
            coeff = fast::sqrt(Scalar(6.0) * T / deltaT_gamma);
            if (d_noiseless_t)
                coeff = Scalar(0.0);
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        brownian_force.x = rx * coeff;
        brownian_force.y = ry * coeff;
        brownian_force.z = rz * coeff;

        // update position

        Scalar mu = 0;

        Scalar inv_alpha = -deltaT_gamma;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 residual;
        Scalar resid;
        unsigned int iteration = 0;
        const unsigned int maxiteration = 10;

        do
            {
            iteration++;
            residual.x = postype.x - next_pos.x
                         + (net_force.x + brownian_force.x - mu * normal.x) * deltaT_gamma;
            residual.y = postype.y - next_pos.y
                         + (net_force.y + brownian_force.y - mu * normal.y) * deltaT_gamma;
            residual.z = postype.z - next_pos.z
                         + (net_force.z + brownian_force.z - mu * normal.z) * deltaT_gamma;
            resid = manifold.implicitFunction(next_pos);

            Scalar3 next_normal = manifold.derivative(next_pos);

            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos.x = next_pos.x - beta * normal.x + residual.x;
            next_pos.y = next_pos.y - beta * normal.y + residual.y;
            next_pos.z = next_pos.z - beta * normal.z + residual.z;
            mu = mu - beta * inv_alpha;

            resid = fabs(resid);
            Scalar vec_norm = sqrt(dot(residual, residual));
            if (vec_norm > resid)
                resid = vec_norm;

            } while (resid > tolerance && iteration < maxiteration);

        net_force.x -= mu * normal.x;
        net_force.y -= mu * normal.y;
        net_force.z -= mu * normal.z;

        virial0 -= mu * normal.x * postype.x;
        virial1 -= 0.5 * mu * (normal.x * postype.y + normal.y * postype.x);
        virial2 -= 0.5 * mu * (normal.x * postype.z + normal.z * postype.x);
        virial3 -= mu * normal.y * postype.y;
        virial4 -= 0.5 * mu * (normal.y * postype.z + normal.z * postype.y);
        virial5 -= mu * normal.z * postype.z;

        d_net_force[idx] = net_force;
        d_net_virial[0 * net_virial_pitch + idx] = virial0;
        d_net_virial[1 * net_virial_pitch + idx] = virial1;
        d_net_virial[2 * net_virial_pitch + idx] = virial2;
        d_net_virial[3 * net_virial_pitch + idx] = virial3;
        d_net_virial[4 * net_virial_pitch + idx] = virial4;
        d_net_virial[5 * net_virial_pitch + idx] = virial5;
        }
    }

template<class Manifold>
hipError_t gpu_include_rattle_force_bd(const Scalar4* d_pos,
                                       Scalar4* d_net_force,
                                       Scalar* d_net_virial,
                                       const unsigned int* d_tag,
                                       const unsigned int* d_group_members,
                                       const unsigned int group_size,
                                       const rattle_bd_step_one_args& rattle_bd_args,
                                       Manifold manifold,
                                       size_t net_virial_pitch,
                                       const Scalar deltaT,
                                       const bool d_noiseless_t,
                                       const GPUPartition& gpu_partition)
    {
    unsigned int run_block_size = 256;

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        const auto shared_bytes
            = (sizeof(Scalar) * rattle_bd_args.n_types + sizeof(Scalar3) * rattle_bd_args.n_types);

        if (shared_bytes > rattle_bd_args.devprop.sharedMemPerBlock)
            {
            throw std::runtime_error("Brownian gamma parameters exceed the available shared "
                                     "memory per block.");
            }

        // run the kernel
        hipLaunchKernelGGL((gpu_include_rattle_force_bd_kernel<Manifold>),
                           dim3(grid),
                           dim3(threads),
                           shared_bytes,
                           0,
                           d_pos,
                           d_net_force,
                           d_net_virial,
                           d_tag,
                           d_group_members,
                           nwork,
                           rattle_bd_args.d_gamma,
                           rattle_bd_args.n_types,
                           rattle_bd_args.timestep,
                           rattle_bd_args.seed,
                           rattle_bd_args.T,
                           rattle_bd_args.tolerance,
                           manifold,
                           net_virial_pitch,
                           deltaT,
                           d_noiseless_t,
                           range.first);
        }

    return hipSuccess;
    }

#endif
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_RATTLE_BD_GPU_CUH__
