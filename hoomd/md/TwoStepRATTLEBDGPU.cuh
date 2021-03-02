#include "hip/hip_runtime.h"
// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepRATTLEBDGPU.cuh
    \brief Declares GPU kernel code for Brownian dynamics on the GPU. Used by TwoStepRATTLEBDGPU.
*/

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/CachedAllocator.h"

#include "hoomd/GPUPartition.cuh"


#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
using namespace hoomd;

#include <assert.h>
#include <type_traits>


#ifndef __TWO_STEP_RATTLE_BD_GPU_CUH__
#define __TWO_STEP_RATTLE_BD_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_rattle_bd_step_one()
struct rattle_bd_step_one_args
    {
    Scalar *d_gamma;          //!< Device array listing per-type gammas
    size_t n_types;          //!< Number of types in \a d_gamma
    bool use_alpha;          //!< Set to true to scale diameters by alpha to get gamma
    Scalar alpha;            //!< Scale factor to convert diameter to alpha
    Scalar T;                 //!< Current temperature
    Scalar eta;
    uint64_t timestep;    //!< Current timestep
    uint16_t seed;            //!< User chosen random number seed
    };


hipError_t gpu_rattle_brownian_step_one(Scalar4 *d_pos,
                                  int3 *d_image,
                                  const BoxDim& box,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  const unsigned int *d_group_members,
                                  const unsigned int group_size,
                                  const Scalar4 *d_net_force,
                                  const Scalar3 *d_f_brownian,
                                  const Scalar3 *d_gamma_r,
                                  Scalar4 *d_orientation,
                                  Scalar4 *d_torque,
                                  const Scalar3 *d_inertia,
                                  Scalar4 *d_angmom,
                                  const rattle_bd_step_one_args& rattle_bd_args,
                                  const bool aniso,
                                  const Scalar deltaT,
                                  const unsigned int D,
                                  const bool d_noiseless_r,
                                  const GPUPartition& gpu_partition
                                  );

template<class Manifold>
hipError_t gpu_include_rattle_force_bd(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar4 *d_net_force,
                                  Scalar3 *d_f_brownian,
                                  Scalar *d_net_virial,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  const unsigned int *d_group_members,
                                  const unsigned int group_size,
                                  const rattle_bd_step_one_args& rattle_bd_args,
			                      Manifold manifold,
                                  size_t net_virial_pitch,
                                  const Scalar deltaT,
                                  const bool d_noiseless_t,
                                  const GPUPartition& gpu_partition
                                  );

#ifdef __HIPCC__


template<class Manifold>
__global__ void gpu_include_rattle_force_bd_kernel(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar4 *d_net_force,
                                  Scalar3 *d_f_brownian,
                                  Scalar *d_net_virial,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  const unsigned int *d_group_members,
                                  const unsigned int nwork,
                                  const Scalar *d_gamma,
                                  const size_t n_types,
                                  const bool use_alpha,
                                  const Scalar alpha,
                                  const uint64_t timestep,
                                  const uint16_t seed,
                                  const Scalar T,
                                  const Scalar eta,
                                  Manifold manifold,
                                  size_t net_virial_pitch,
                                  const Scalar deltaT,
                                  const bool d_noiseless_t,
                                  const unsigned int offset)
    {
    HIP_DYNAMIC_SHARED( char, s_data2)

    Scalar3 *s_gammas_r = (Scalar3 *)s_data2;
    Scalar *s_gammas = (Scalar *)(s_gammas_r + n_types);

    if (!use_alpha)
        {
        // read in the gamma (1 dimensional array), stored in s_gammas[0: n_type] (Pythonic convention)
        for (int cur_offset = 0; cur_offset < n_types; cur_offset += blockDim.x)
            {
            if (cur_offset + threadIdx.x < n_types)
                s_gammas[cur_offset + threadIdx.x] = d_gamma[cur_offset + threadIdx.x];
            }
        __syncthreads();
        }

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
        Scalar3 brownian_force = d_f_brownian[tag];

        Scalar virial0 = d_net_virial[0*net_virial_pitch+idx];
        Scalar virial1 = d_net_virial[1*net_virial_pitch+idx];
        Scalar virial2 = d_net_virial[2*net_virial_pitch+idx];
        Scalar virial3 = d_net_virial[3*net_virial_pitch+idx];
        Scalar virial4 = d_net_virial[4*net_virial_pitch+idx];
        Scalar virial5 = d_net_virial[5*net_virial_pitch+idx];

        // calculate the magnitude of the random force
        Scalar gamma;
        if (use_alpha)
            {
            // determine gamma from diameter
            gamma = alpha*d_diameter[idx];
            }
        else
            {
            // determine gamma from type
            unsigned int typ = __scalar_as_int(postype.w);
            gamma = s_gammas[typ];
            }
        Scalar deltaT_gamma = deltaT/gamma;


        // compute the random force
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),
                            hoomd::Counter(tag, 2));


	Scalar3 next_pos;
	next_pos.x = postype.x;
	next_pos.y = postype.y;
	next_pos.z = postype.z;
        Scalar3 normal = manifold.derivative(next_pos);

        // draw a new random velocity for particle j
        Scalar mass = vel.w;
        Scalar sigma = fast::sqrt(T/mass);
        NormalDistribution<Scalar> norm(sigma);
        vel.x = norm(rng);
        vel.y = norm(rng);
        vel.z = norm(rng);

        Scalar norm_normal = 1.0/fast::sqrt(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);

        normal.x = norm_normal*normal.x;
        normal.y = norm_normal*normal.y;
        normal.z = norm_normal*normal.z;

        Scalar rand_norm = vel.x*normal.x+ vel.y*normal.y + vel.z*normal.z;
        vel.x -= rand_norm*normal.x;
        vel.y -= rand_norm*normal.y;
        vel.z -= rand_norm*normal.z;

        Scalar rx, ry, rz, coeff;

	    if (T > 0)
	        {
	    	UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
	    	rx = uniform(rng);
	    	ry = uniform(rng);
	    	rz = uniform(rng);

	    	Scalar3 proj = normal;
	    	Scalar proj_norm = 1.0/fast::sqrt(proj.x*proj.x+proj.y*proj.y+proj.z*proj.z);
	    	proj.x *= proj_norm;
	    	proj.y *= proj_norm;
	    	proj.z *= proj_norm;

	    	Scalar proj_r = rx*proj.x + ry*proj.y + rz*proj.z;

	    	rx = rx - proj_r*proj.x;
	    	ry = ry - proj_r*proj.y;
	    	rz = rz - proj_r*proj.z;

                    // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
                    // it is not the dimensionality of the system
                    coeff = fast::sqrt(Scalar(6.0)*T/deltaT_gamma);
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


            brownian_force.x = rx*coeff;
            brownian_force.y = ry*coeff;
            brownian_force.z = rz*coeff;

            // update position

	        Scalar mu = 0;

                unsigned int maxiteration = 10;
	        Scalar inv_alpha = -deltaT_gamma;
	        inv_alpha = Scalar(1.0)/inv_alpha;

	        Scalar3 residual;
	        Scalar resid;
	        unsigned int iteration = 0;

	        do
	        {
	            iteration++;
	            residual.x = postype.x - next_pos.x + (net_force.x + brownian_force.x - mu*normal.x) * deltaT_gamma;
	            residual.y = postype.y - next_pos.y + (net_force.y + brownian_force.y - mu*normal.y) * deltaT_gamma;
	            residual.z = postype.z - next_pos.z + (net_force.z + brownian_force.z - mu*normal.z) * deltaT_gamma;
	            resid = manifold.implicit_function(next_pos);

                    Scalar3 next_normal =  manifold.derivative(next_pos);


	            Scalar nndotr = dot(next_normal,residual);
	            Scalar nndotn = dot(next_normal,normal);
	            Scalar beta = (resid + nndotr)/nndotn;

                    next_pos.x = next_pos.x - beta*normal.x + residual.x;
                    next_pos.y = next_pos.y - beta*normal.y + residual.y;
                    next_pos.z = next_pos.z - beta*normal.z + residual.z;
	            mu = mu - beta*inv_alpha;

	            resid = fabs(resid);
                    Scalar vec_norm = sqrt(dot(residual,residual));
                    if ( vec_norm > resid) resid =  vec_norm;

	        //} while (maxNormGPU(residual,resid) > eta && iteration < maxiteration );
	        } while (resid > eta && iteration < maxiteration );

            net_force.x -= mu*normal.x;
            net_force.y -= mu*normal.y;
            net_force.z -= mu*normal.z;

        virial0 -= mu*normal.x*postype.x;
        virial1 -= 0.5*mu*(normal.x*postype.y+normal.y*postype.x);
        virial2 -= 0.5*mu*(normal.x*postype.z+normal.z*postype.x);
        virial3 -= mu*normal.y*postype.y;
        virial4 -= 0.5*mu*(normal.y*postype.z+normal.z*postype.y);
        virial5 -= mu*normal.z*postype.z;

        d_f_brownian[tag] = brownian_force;

        d_net_force[idx] = net_force;
        d_net_virial[0*net_virial_pitch+idx] = virial0;
        d_net_virial[1*net_virial_pitch+idx] = virial1;
        d_net_virial[2*net_virial_pitch+idx] = virial2;
        d_net_virial[3*net_virial_pitch+idx] = virial3;
        d_net_virial[4*net_virial_pitch+idx] = virial4;
        d_net_virial[5*net_virial_pitch+idx] = virial5;

        d_vel[idx] = vel;

        }
    }

template<class Manifold>
hipError_t gpu_include_rattle_force_bd(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar4 *d_net_force,
                                  Scalar3 *d_f_brownian,
                                  Scalar *d_net_virial,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  const unsigned int *d_group_members,
                                  const unsigned int group_size,
                                  const rattle_bd_step_one_args& rattle_bd_args,
				                  Manifold manifold,
                                  size_t net_virial_pitch,
                                  const Scalar deltaT,
                                  const bool d_noiseless_t,
                                  const GPUPartition& gpu_partition
                                  )
    {
    unsigned int run_block_size = 256;

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid( (nwork/run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_include_rattle_force_bd_kernel<Manifold>), dim3(grid), dim3(threads), (unsigned int)(sizeof(Scalar)*rattle_bd_args.n_types + sizeof(Scalar3)*rattle_bd_args.n_types), 0, d_pos,
                                     d_vel,
                                     d_net_force,
                                     d_f_brownian,
                                     d_net_virial,
                                     d_diameter,
                                     d_tag,
                                     d_group_members,
                                     nwork,
                                     rattle_bd_args.d_gamma,
                                     rattle_bd_args.n_types,
                                     rattle_bd_args.use_alpha,
                                     rattle_bd_args.alpha,
                                     rattle_bd_args.timestep,
                                     rattle_bd_args.seed,
                                     rattle_bd_args.T,
                                     rattle_bd_args.eta,
				                     manifold,
                                     net_virial_pitch,
                                     deltaT,
                                     d_noiseless_t,
                                     range.first);
        }

    return hipSuccess;
    }

#endif
#endif //__TWO_STEP_RATTLE_BD_GPU_CUH__
