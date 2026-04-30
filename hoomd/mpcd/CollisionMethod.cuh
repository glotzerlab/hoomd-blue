// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.cuh
 * \brief Declaration of CUDA kernels for mpcd::CollisionMethod
 */

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {

cudaError_t draw_velocities_constituent_particles(Scalar3* d_linmom_accum,
                                                  Scalar3* d_angmom_accum,
                                                  Scalar4* d_alt_vel,
                                                  const Scalar4* d_postype,
                                                  const Scalar4* d_velocity,
                                                  const int3* d_image,
                                                  const unsigned int* d_tag,
                                                  const unsigned int* d_lookup_center,
                                                  const BoxDim& global_box,
                                                  const uint64_t timestep,
                                                  const uint16_t seed,
                                                  const Scalar T,
                                                  const unsigned int num_total,
                                                  const unsigned int block_size);

cudaError_t get_net_velocity_rigid_body(const Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_alt_vel,
                                        const Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_rigid_center,
                                        const unsigned int num_centers,
                                        const unsigned int block_size);

cudaError_t apply_thermalized_velocity_vectors(const Scalar3* d_angmom_accum,
                                               const Scalar4* d_alt_vel,
                                               const Scalar4* d_postype,
                                               Scalar4* d_velocity,
                                               const int3* d_image,
                                               const unsigned int* d_lookup_center,
                                               const BoxDim& global_box,
                                               const unsigned int num_total,
                                               const unsigned int block_size);

cudaError_t store_initial_embedded_group_velocities(Scalar4* d_initial_vel,
                                                    const Scalar4* d_velocity,
                                                    const unsigned int* d_embed_group,
                                                    const unsigned int num_group,
                                                    const unsigned int block_size);

cudaError_t accumulate_rigid_body_momenta(Scalar3* d_linmom_accum,
                                          Scalar3* d_angmom_accum,
                                          const Scalar4* d_initial_vel,
                                          const unsigned int* d_embed_group,
                                          const Scalar4* d_postype,
                                          const Scalar4* d_velocity,
                                          const int3* d_image,
                                          const unsigned int* d_lookup_center,
                                          const BoxDim& global_box,
                                          const unsigned int num_group,
                                          const unsigned int block_size);

cudaError_t transfer_rigid_body_momenta(Scalar3* d_linmom_accum,
                                        Scalar3* d_angmom_accum,
                                        Scalar4* d_velocity,
                                        const Scalar4* d_orientation,
                                        Scalar4* d_angmom,
                                        const Scalar3* d_inertia,
                                        const unsigned int* d_rigid_center,
                                        const unsigned int num_centers,
                                        const unsigned int block_size);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
