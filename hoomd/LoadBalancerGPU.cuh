// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file LoadBalancerGPU.cuh
    \brief Defines the GPU functions for load balancing
*/

#ifdef ENABLE_MPI
#include "ParticleData.cuh"
#include "Index1D.h"

//! Kernel drive to mark the current rank of each particle
void gpu_load_balance_mark_rank(unsigned int *d_ranks,
                                const Scalar4 *d_pos,
                                const unsigned int *d_cart_ranks,
                                const uint3 rank_pos,
                                const BoxDim& box,
                                const Index3D& di,
                                const unsigned int N,
                                const unsigned int block_size);

//! thrust driver to select the particles that are off rank
unsigned int gpu_load_balance_select_off_rank(unsigned int *d_off_rank,
                                              unsigned int *d_ranks,
                                              const unsigned int N,
                                              const unsigned int cur_rank);
#endif // ENABLE_MPI
