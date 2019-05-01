// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ParticleData.cuh
 * \brief Declares GPU functions used by mpcd::ParticleData
 */

#ifndef MPCD_PARTICLE_DATA_CUH_
#define MPCD_PARTICLE_DATA_CUH_

#include <cuda_runtime.h>

#ifdef ENABLE_MPI
#include "hoomd/BoxDim.h"
#include "ParticleDataUtilities.h"

namespace mpcd
{
namespace gpu
{
//! Marks the particles which are being removed
cudaError_t mark_removed_particles(unsigned char *d_remove_flags,
                                   const unsigned int *d_comm_flags,
                                   const unsigned int mask,
                                   const unsigned int N,
                                   const unsigned int block_size);

//! Partition the indexes of particles to keep or remove
cudaError_t partition_particles(void *d_tmp,
                                size_t& tmp_bytes,
                                const unsigned char *d_remove_flags,
                                unsigned int *d_remove_ids,
                                unsigned int *d_num_remove,
                                const unsigned int N);

//! Pack particle data into output buffer and remove marked particles
cudaError_t remove_particles(mpcd::detail::pdata_element *d_out,
                             Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             unsigned int *d_tag,
                             unsigned int *d_comm_flags,
                             unsigned int *d_remove_ids,
                             const unsigned int n_remove,
                             const unsigned int N,
                             const unsigned int block_size);

//! Update particle data with new particles
void add_particles(unsigned int old_nparticles,
                   unsigned int num_add_ptls,
                   Scalar4 *d_pos,
                   Scalar4 *d_vel,
                   unsigned int *d_tag,
                   unsigned int *d_comm_flags,
                   const mpcd::detail::pdata_element *d_in,
                   const unsigned int mask,
                   const unsigned int block_size);
} // end namespace gpu
} // end namespace mpcd

#endif // ENABLE_MPI

#endif // MPCD_PARTICLE_DATA_CUH_
