// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_AT_COLLISION_METHOD_GPU_CUH_
#define MPCD_AT_COLLISION_METHOD_GPU_CUH_

/*!
 * \file mpcd/ATCollisionMethodGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::ATCollisionMethodGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Draw particle velocities for the Andersen thermostat from Gaussian distribution
cudaError_t at_draw_velocity(Scalar4* d_alt_vel,
                             Scalar4* d_alt_vel_embed,
                             const unsigned int* d_tag,
                             const Scalar mpcd_mass,
                             const unsigned int* d_embed_idx,
                             const Scalar4* d_vel_embed,
                             const unsigned int* d_tag_embed,
                             const uint64_t timestep,
                             const uint16_t seed,
                             const Scalar T,
                             const unsigned int N_mpcd,
                             const unsigned int N_tot,
                             const unsigned int block_size);

//! Apply velocities for the Andersen thermostat
cudaError_t at_apply_velocity(Scalar4* d_vel,
                              Scalar4* d_vel_embed,
                              const Scalar4* d_vel_alt,
                              const unsigned int* d_embed_idx,
                              const Scalar4* d_vel_alt_embed,
                              const unsigned int* d_embed_cell_ids,
                              const double4* d_cell_vel,
                              const double4* d_rand_vel,
                              const unsigned int N_mpcd,
                              const unsigned int N_tot,
                              const unsigned int block_size);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_AT_COLLISION_METHOD_GPU_CUH_
