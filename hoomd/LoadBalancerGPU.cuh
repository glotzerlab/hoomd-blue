// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancerGPU.cuh
    \brief Defines the GPU functions for load balancing
*/

#ifdef ENABLE_MPI

#include "Index1D.h"
#include "ParticleData.cuh"

namespace hoomd
    {
namespace kernel
    {
//! Kernel drive to mark the current rank of each particle
void gpu_load_balance_mark_rank(unsigned int* d_ranks,
                                const Scalar4* d_pos,
                                const unsigned int* d_cart_ranks,
                                const uint3 rank_pos,
                                const BoxDim& box,
                                const Index3D& di,
                                const unsigned int N,
                                const unsigned int block_size);

//! thrust driver to select the particles that are off rank
unsigned int gpu_load_balance_select_off_rank(unsigned int* d_off_rank,
                                              unsigned int* d_ranks,
                                              const unsigned int N,
                                              const unsigned int cur_rank);

    } // end namespace kernel

    } // end namespace hoomd

#endif // ENABLE_MPI
