// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_PARTICLE_DATA_CUH_
#define MPCD_PARTICLE_DATA_CUH_

#include <cuda_runtime.h>
#include "hoomd/extern/util/mgpucontext.h"

#ifdef ENABLE_MPI
#include "hoomd/BoxDim.h"
#include "ParticleDataUtilities.h"

/*!
 * \file mpcd/ParticleData.cuh
 * \brief Declares GPU functions used by mpcd::ParticleData
 */

namespace mpcd
{
namespace gpu
{
//! Pack particle data into output buffer and remove marked particles
unsigned int remove_particles(unsigned int N,
                              const Scalar4 *d_pos,
                              const Scalar4 *d_vel,
                              const unsigned int *d_tag,
                              Scalar4 *d_pos_alt,
                              Scalar4 *d_vel_alt,
                              unsigned int *d_tag_alt,
                              mpcd::detail::pdata_element *d_out,
                              unsigned int *d_comm_flags,
                              unsigned int *d_comm_flags_out,
                              unsigned int max_n_out,
                              unsigned int *d_tmp,
                              mgpu::ContextPtr mgpu_context);

//! Update particle data with new particles
void add_particles(unsigned int old_nparticles,
                   unsigned int num_add_ptls,
                   Scalar4 *d_pos,
                   Scalar4 *d_vel,
                   unsigned int *d_tag,
                   const mpcd::detail::pdata_element *d_in,
                   unsigned int *d_comm_flags);
} // end namespace gpu
} // end namespace mpcd

#endif // ENABLE_MPI

#endif // MPCD_PARTICLE_DATA_CUH_
