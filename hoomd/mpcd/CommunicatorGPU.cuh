// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CommunicatorGPU.cuh
 * \brief Defines the GPU functions of the communication algorithms
 */

#ifdef ENABLE_MPI
#include "ParticleDataUtilities.h"
#include "hoomd/BoxDim.h"
#include "hoomd/Index1D.h"

namespace mpcd
{
namespace gpu
{
//! Mark particles that have left the local box for sending
cudaError_t stage_particles(unsigned int *d_comm_flag,
                            const Scalar4 *d_pos,
                            const unsigned int n,
                            const BoxDim& box,
                            const unsigned int block_size);

//! Sort the particle send buffer on the GPU
size_t sort_comm_send_buffer(mpcd::detail::pdata_element *d_sendbuf,
                             unsigned int *d_neigh_send,
                             unsigned int *d_num_send,
                             unsigned int *d_tmp_keys,
                             const uint3 grid_pos,
                             const Index3D& di,
                             const unsigned int mask,
                             const unsigned int *d_cart_ranks,
                             const unsigned int Nsend);

//! Reduce communication flags with bitwise OR using the CUB library
void reduce_comm_flags(unsigned int *d_req_flags,
                       void *d_tmp,
                       size_t& tmp_bytes,
                       const unsigned int *d_comm_flags,
                       const unsigned int N);

//! Apply boundary conditions
void wrap_particles(const unsigned int n_recv,
                    mpcd::detail::pdata_element *d_in,
                    const BoxDim& box);
} // end namespace gpu
} // end namespace mpcd

#endif // ENABLE_MPI
