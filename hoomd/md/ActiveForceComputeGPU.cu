// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceComputeGPU.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file ActiveForceComputeGPU.cu
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by
   ActiveForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for setting active force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_force particle force on device
    \param d_torque particle torque on device
    \param d_orientation particle orientation on device
    \param d_f_act particle active force unit vector
    \param d_t_act particle active torque unit vector
    \param orientationLink check if particle orientation is linked to active force vector
*/
__global__ void gpu_compute_active_force_set_forces_kernel(const unsigned int group_size,
                                                           unsigned int* d_index_array,
                                                           Scalar4* d_force,
                                                           Scalar4* d_torque,
                                                           const Scalar4* d_pos,
                                                           const Scalar4* d_orientation,
                                                           const Scalar4* d_f_act,
                                                           const Scalar4* d_t_act,
                                                           const unsigned int N)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    Scalar4 fact = __ldg(d_f_act + type);

    vec3<Scalar> f(fact.w * fact.x, fact.w * fact.y, fact.w * fact.z);
    quat<Scalar> quati(__ldg(d_orientation + idx));
    vec3<Scalar> fi = rotate(quati, f);
    d_force[idx] = vec_to_scalar4(fi, 0);

    Scalar4 tact = __ldg(d_t_act + type);

    vec3<Scalar> t(tact.w * tact.x, tact.w * tact.y, tact.w * tact.z);
    vec3<Scalar> ti = rotate(quati, t);
    d_torque[idx] = vec_to_scalar4(ti, 0);
    }

//! Kernel for applying rotational diffusion to active force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_act particle active force unit vector
    \param is2D check if simulation is 2D or 3D
    \param rotationConst particle rotational diffusion constant
    \param seed seed for random number generator
*/
__global__ void gpu_compute_active_force_rotational_diffusion_kernel(const unsigned int group_size,
                                                                     unsigned int* d_tag,
                                                                     unsigned int* d_index_array,
                                                                     const Scalar4* d_pos,
                                                                     Scalar4* d_orientation,
                                                                     const Scalar4* d_f_act,
                                                                     bool is2D,
                                                                     const Scalar rotationConst,
                                                                     const uint64_t timestep,
                                                                     const uint16_t seed)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    Scalar4 fact = __ldg(d_f_act + type);

    if (fact.w != 0)
        {
        unsigned int ptag = d_tag[group_idx];

        quat<Scalar> quati(__ldg(d_orientation + idx));

        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::ActiveForceCompute, timestep, seed),
            hoomd::Counter(ptag));

        if (is2D) // 2D
            {
            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationConst)(rng);

            vec3<Scalar> b(0, 0, 1.0);
            quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(b, delta_theta);

            quati = rot_quat * quati;
            quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
            d_orientation[idx] = quat_to_scalar4(quati);
            // in 2D there is only one meaningful direction for torque
            }
        else // 3D: Following Stenhammar, Soft Matter, 2014
            {
            hoomd::SpherePointGenerator<Scalar> unit_vec;
            vec3<Scalar> rand_vec;
            unit_vec(rng, rand_vec);

            vec3<Scalar> f(fact.x, fact.y, fact.z);
            vec3<Scalar> fi = rotate(quati, f);

            vec3<Scalar> aux_vec = cross(fi, rand_vec); // rotation axis
            Scalar aux_vec_mag = slow::rsqrt(dot(aux_vec, aux_vec));
            aux_vec *= aux_vec_mag;

            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationConst)(rng);
            quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(aux_vec, delta_theta);

            quati = rot_quat * quati;
            quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
            d_orientation[idx] = quat_to_scalar4(quati);
            }
        }
    }

hipError_t gpu_compute_active_force_set_forces(const unsigned int group_size,
                                               unsigned int* d_index_array,
                                               Scalar4* d_force,
                                               Scalar4* d_torque,
                                               const Scalar4* d_pos,
                                               const Scalar4* d_orientation,
                                               const Scalar4* d_f_act,
                                               const Scalar4* d_t_act,
                                               const unsigned int N,
                                               unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipMemset(d_force, 0, sizeof(Scalar4) * N);
    hipLaunchKernelGGL((gpu_compute_active_force_set_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_index_array,
                       d_force,
                       d_torque,
                       d_pos,
                       d_orientation,
                       d_f_act,
                       d_t_act,
                       N);
    return hipSuccess;
    }

hipError_t gpu_compute_active_force_rotational_diffusion(const unsigned int group_size,
                                                         unsigned int* d_tag,
                                                         unsigned int* d_index_array,
                                                         const Scalar4* d_pos,
                                                         Scalar4* d_orientation,
                                                         const Scalar4* d_f_act,
                                                         bool is2D,
                                                         const Scalar rotationConst,
                                                         const uint64_t timestep,
                                                         const uint16_t seed,
                                                         unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_active_force_rotational_diffusion_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_tag,
                       d_index_array,
                       d_pos,
                       d_orientation,
                       d_f_act,
                       is2D,
                       rotationConst,
                       timestep,
                       seed);
    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
