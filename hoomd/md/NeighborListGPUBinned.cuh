// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#ifndef __NEIGHBORLOSTGPUBINNED_CUH__
#define __NEIGHBORLOSTGPUBINNED_CUH__

#include <hip/hip_runtime.h>

#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file NeighborListGPUBinned.cuh
    \brief Declares GPU kernel code for neighbor list generation on the GPU
*/

// currently this is hardcoded, we should set it to the max of platforms
#if defined(__HIP_PLATFORM_NVCC__)
#define WARP_SIZE 32
#elif defined(__HIP_PLATFORM_HCC__)
#define WARP_SIZE 64
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
const unsigned int min_threads_per_particle = 1;
const unsigned int max_threads_per_particle = WARP_SIZE;

//! Kernel driver for gpu_compute_nlist_kernel()
hipError_t gpu_compute_nlist_binned(unsigned int* d_nlist,
                                    unsigned int* d_n_neigh,
                                    Scalar4* d_last_updated_pos,
                                    unsigned int* d_conditions,
                                    const unsigned int* d_Nmax,
                                    const unsigned int* d_head_list,
                                    const Scalar4* d_pos,
                                    const unsigned int* d_body,
                                    const Scalar* d_diameter,
                                    const unsigned int N,
                                    const unsigned int* d_cell_size,
                                    const Scalar4* d_cell_xyzf,
                                    const unsigned int* d_cell_idx,
                                    const Scalar4* d_cell_tdb,
                                    const unsigned int* d_cell_adj,
                                    const Index3D& ci,
                                    const Index2D& cli,
                                    const Index2D& cadji,
                                    const BoxDim& box,
                                    const Scalar* d_r_cut,
                                    const Scalar r_buff,
                                    const unsigned int ntypes,
                                    const unsigned int threads_per_particle,
                                    const unsigned int block_size,
                                    bool filter_body,
                                    bool diameter_shift,
                                    const Scalar3& ghost_width,
                                    const GPUPartition& gpu_partition,
                                    bool use_index);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
