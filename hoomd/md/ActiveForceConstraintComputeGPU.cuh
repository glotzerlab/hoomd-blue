// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/TextureTools.h"

/*! \file ActiveForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating active forces forces on the GPU. Used by
   ActiveForceComputeGPU.
*/

#ifndef __ACTIVE_FORCE_CONSTRAINT_COMPUTE_GPU_CUH__
#define __ACTIVE_FORCE_CONSTRAINT_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template<class Manifold>
hipError_t gpu_compute_active_force_set_constraints(const unsigned int group_size,
                                                    unsigned int* d_index_array,
                                                    const Scalar4* d_pos,
                                                    Scalar4* d_orientation,
                                                    const Scalar4* d_f_act,
                                                    Manifold manifold,
                                                    unsigned int block_size);

template<class Manifold>
hipError_t gpu_compute_active_force_constraint_rotational_diffusion(const unsigned int group_size,
                                                                    unsigned int* d_tag,
                                                                    unsigned int* d_index_array,
                                                                    const Scalar4* d_pos,
                                                                    Scalar4* d_orientation,
                                                                    Manifold manifold,
                                                                    bool is2D,
                                                                    const Scalar rotationDiff,
                                                                    const uint64_t timestep,
                                                                    const uint16_t seed,
                                                                    unsigned int block_size);

#ifdef __HIPCC__

//! Kernel for adjusting active force vectors to align parallel to an
//  manifold surface constraint on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_act particle active force unit vector
    \param manifold constraint
*/
template<class Manifold>
__global__ void gpu_compute_active_force_set_constraints_kernel(const unsigned int group_size,
                                                                unsigned int* d_index_array,
                                                                const Scalar4* d_pos,
                                                                Scalar4* d_orientation,
                                                                const Scalar4* d_f_act,
                                                                Manifold manifold)
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
        Scalar3 current_pos = make_scalar3(posidx.x, posidx.y, posidx.z);

        vec3<Scalar> norm = normalize(vec3<Scalar>(manifold.derivative(current_pos)));

        vec3<Scalar> f(fact.x, fact.y, fact.z);
        quat<Scalar> quati(__ldg(d_orientation + idx));
        vec3<Scalar> fi = rotate(quati, f);

        Scalar dot_prod = fi.x * norm.x + fi.y * norm.y + fi.z * norm.z;

        Scalar dot_perp_prod = slow::rsqrt(1 - dot_prod * dot_prod);

        Scalar phi = slow::atan(dot_prod * dot_perp_prod);

        fi.x -= norm.x * dot_prod;
        fi.y -= norm.y * dot_prod;
        fi.z -= norm.z * dot_prod;

        Scalar new_norm = slow::rsqrt(fi.x * fi.x + fi.y * fi.y + fi.z * fi.z);

        fi *= new_norm;

        vec3<Scalar> rot_vec = cross(norm, fi);

        quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(rot_vec, phi);

        quati = rot_quat * quati;
        quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
        d_orientation[idx] = quat_to_scalar4(quati);
        }
    }

//! Kernel for applying rotational diffusion to active force vectors on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param manifold constraint
    \param is2D check if simulation is 2D or 3D
    \param rotationConst particle rotational diffusion constant
    \param seed seed for random number generator
*/
template<class Manifold>
__global__ void
gpu_compute_active_force_constraint_rotational_diffusion_kernel(const unsigned int group_size,
                                                                unsigned int* d_tag,
                                                                unsigned int* d_index_array,
                                                                const Scalar4* d_pos,
                                                                Scalar4* d_orientation,
                                                                Manifold manifold,
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
    unsigned int ptag = d_tag[group_idx];

    quat<Scalar> quati(__ldg(d_orientation + idx));

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::ActiveForceCompute, timestep, seed),
        hoomd::Counter(ptag));

    Scalar3 current_pos = make_scalar3(posidx.x, posidx.y, posidx.z);
    vec3<Scalar> norm = normalize(vec3<Scalar>(manifold.derivative(current_pos)));

    Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationConst)(rng);

    quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(norm, delta_theta);

    quati = rot_quat * quati;
    quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
    d_orientation[idx] = quat_to_scalar4(quati);
    }

template<class Manifold>
hipError_t gpu_compute_active_force_set_constraints(const unsigned int group_size,
                                                    unsigned int* d_index_array,
                                                    const Scalar4* d_pos,
                                                    Scalar4* d_orientation,
                                                    const Scalar4* d_f_act,
                                                    Manifold manifold,
                                                    unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_active_force_set_constraints_kernel<Manifold>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_index_array,
                       d_pos,
                       d_orientation,
                       d_f_act,
                       manifold);
    return hipSuccess;
    }

template<class Manifold>
hipError_t gpu_compute_active_force_constraint_rotational_diffusion(const unsigned int group_size,
                                                                    unsigned int* d_tag,
                                                                    unsigned int* d_index_array,
                                                                    const Scalar4* d_pos,
                                                                    Scalar4* d_orientation,
                                                                    Manifold manifold,
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
    hipLaunchKernelGGL((gpu_compute_active_force_constraint_rotational_diffusion_kernel<Manifold>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
                       d_tag,
                       d_index_array,
                       d_pos,
                       d_orientation,
                       manifold,
                       is2D,
                       rotationConst,
                       timestep,
                       seed);
    return hipSuccess;
    }

#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
