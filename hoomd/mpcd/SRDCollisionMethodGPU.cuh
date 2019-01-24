// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_SRD_COLLISION_METHOD_GPU_CUH_
#define MPCD_SRD_COLLISION_METHOD_GPU_CUH_

/*!
 * \file mpcd/SRDCollisionMethodGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SRDCollisionMethodGPU
 */

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace mpcd
{
namespace gpu
{

cudaError_t srd_draw_vectors(double3 *d_rotvec,
                             double *d_factors,
                             const double3 *d_cell_energy,
                             const Index3D& ci,
                             const int3 origin,
                             const uint3 global_dim,
                             const Index3D& global_ci,
                             const unsigned int timestep,
                             const unsigned int seed,
                             const Scalar T_set,
                             const unsigned int n_dimensions,
                             const unsigned int block_size);

cudaError_t srd_rotate(Scalar4 *d_vel,
                       Scalar4 *d_vel_embed,
                       const unsigned int *d_embed_group,
                       const unsigned int *d_embed_cell_ids,
                       const double4 *d_cell_vel,
                       const double3 *d_rotvec,
                       const double angle,
                       const double *d_factors,
                       const unsigned int N_mpcd,
                       const unsigned int N_tot,
                       const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_SRD_COLLISION_METHOD_GPU_CUH_
