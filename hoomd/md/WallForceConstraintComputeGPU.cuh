// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/TextureTools.h"

/*! \file WallForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating wall forces forces on the GPU. Used by
   WallForceComputeGPU.
*/

#ifndef __WALL_FORCE_CONSTRAINT_COMPUTE_GPU_CUH__
#define __WALL_FORCE_CONSTRAINT_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template<class Manifold>
hipError_t gpu_compute_wall_constraint(const unsigned int group_size,
					    unsigned int* d_index_array,
					    Scalar4* d_force,
					    Scalar* d_virial,
					    const Scalar4* d_pos,
					    Manifold manifold,
					    unsigned int block_size);

template<class Manifold>
hipError_t gpu_compute_wall_friction(const unsigned int group_size,
					    unsigned int* d_index_array,
					    Scalar4* d_force,
					    Scalar* d_virial,
					    const Scalar4* d_pos,
					    const Scalar4* d_net_force,
					    Manifold manifold,
					    bool brownian,
					    unsigned int block_size);

#ifdef __HIPCC__

//! Kernel for adjusting wall force vectors to align parallel to an
//  manifold surface constraint on the GPU
/*! \param group_size number of particles
    \param d_index_array stores list to convert group index to global tag
    \param d_pos particle positions on device
    \param d_f_act particle wall force unit vector
    \param manifold constraint
*/
template<class Manifold>
__global__ void gpu_compute_wall_constraint_kernel(const unsigned int group_size,
						    unsigned int* d_index_array,
						    Scalar4* d_force,
						    Scalar* d_virial,
						    const Scalar4* d_pos,
						    Manifold manifold)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);

    //Scalar4 fact = __ldg(d_f_act + type);

    //if (fact.w != 0)
    //    {
    //    Scalar3 current_pos = make_scalar3(posidx.x, posidx.y, posidx.z);

    //    vec3<Scalar> norm = normalize(vec3<Scalar>(manifold.derivative(current_pos)));

    //    vec3<Scalar> f(fact.x, fact.y, fact.z);
    //    quat<Scalar> quati(__ldg(d_orientation + idx));
    //    vec3<Scalar> fi = rotate(quati, f);

    //    Scalar dot_prod = fi.x * norm.x + fi.y * norm.y + fi.z * norm.z;

    //    Scalar dot_perp_prod = slow::rsqrt(1 - dot_prod * dot_prod);

    //    Scalar phi = slow::atan(dot_prod * dot_perp_prod);

    //    fi.x -= norm.x * dot_prod;
    //    fi.y -= norm.y * dot_prod;
    //    fi.z -= norm.z * dot_prod;

    //    Scalar new_norm = slow::rsqrt(fi.x * fi.x + fi.y * fi.y + fi.z * fi.z);

    //    fi *= new_norm;

    //    vec3<Scalar> rot_vec = cross(norm, fi);

    //    quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(rot_vec, phi);

    //    quati = rot_quat * quati;
    //    quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
    //    d_orientation[idx] = quat_to_scalar4(quati);
    //    }
    }

//! Kernel for applying rotational diffusion to wall force vectors on the GPU
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
gpu_compute_wall_friction_kernel(const unsigned int group_size,
				    unsigned int* d_index_array,
				    Scalar4* d_force,
				    Scalar* d_virial,
				    const Scalar4* d_pos,
				    const Scalar4* d_net_force,
				    Manifold manifold,
				    bool brownian)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= group_size)
        return;

    unsigned int idx = d_index_array[group_idx];
    Scalar4 posidx = __ldg(d_pos + idx);
    unsigned int type = __scalar_as_int(posidx.w);
    //unsigned int ptag = d_tag[group_idx];

    //quat<Scalar> quati(__ldg(d_orientation + idx));

    //hoomd::RandomGenerator rng(
    //    hoomd::Seed(hoomd::RNGIdentifier::WallForceCompute, timestep, seed),
    //    hoomd::Counter(ptag));

    //Scalar3 current_pos = make_scalar3(posidx.x, posidx.y, posidx.z);
    //vec3<Scalar> norm = normalize(vec3<Scalar>(manifold.derivative(current_pos)));

    //Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotationConst)(rng);

    //quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(norm, delta_theta);

    //quati = rot_quat * quati;
    //quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
    //d_orientation[idx] = quat_to_scalar4(quati);
    }

template<class Manifold>
hipError_t gpu_compute_wall_constraint(const unsigned int group_size,
					    unsigned int* d_index_array,
					    Scalar4* d_force,
					    Scalar* d_virial,
					    const Scalar4* d_pos,
					    Manifold manifold,
					    unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_wall_constraint_kernel<Manifold>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
		       d_index_array,
		       d_force,
		       d_virial,
		       d_pos,
                       manifold);
    return hipSuccess;
    }

template<class Manifold>
hipError_t gpu_compute_wall_friction(const unsigned int group_size,
					    unsigned int* d_index_array,
					    Scalar4* d_force,
					    Scalar* d_virial,
					    const Scalar4* d_pos,
					    const Scalar4* d_net_force,
					    Manifold manifold,
					    bool brownian,
					    unsigned int block_size)
    {
    // setup the grid to run the kernel
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_wall_friction_kernel<Manifold>),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       group_size,
		       d_index_array,
		       d_force,
		       d_virial,
		       d_pos,
		       d_net_force,
		       manifold,
		       brownian);

    return hipSuccess;
    }

#endif

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
