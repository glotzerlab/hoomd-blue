// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __NEIGHBORLOSTGPUSTENCIL_CUH__
#define __NEIGHBORLOSTGPUSTENCIL_CUH__

#include <hip/hip_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file NeighborListGPUStencil.cuh
    \brief Declares GPU kernel code for neighbor list generation on the GPU
*/

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

//! Kernel driver for gpu_compute_nlist_multi_binned_kernel()
hipError_t gpu_compute_nlist_stencil(unsigned int* d_nlist,
                                     unsigned int* d_n_neigh,
                                     Scalar4* d_last_updated_pos,
                                     unsigned int* d_conditions,
                                     const unsigned int* d_Nmax,
                                     const size_t* d_head_list,
                                     const unsigned int* d_pid_map,
                                     const Scalar4* d_pos,
                                     const unsigned int* d_body,
                                     const unsigned int N,
                                     const unsigned int* d_cell_size,
                                     const Scalar4* d_cell_xyzf,
                                     const uint2* d_cell_type_body,
                                     const Index3D& ci,
                                     const Index2D& cli,
                                     const Scalar4* d_stencil,
                                     const unsigned int* d_n_stencil,
                                     const Index2D& stencil_idx,
                                     const BoxDim& box,
                                     const Scalar* d_r_cut,
                                     const Scalar r_buff,
                                     const unsigned int ntypes,
                                     const Scalar3& ghost_width,
                                     bool filter_body,
                                     const unsigned int threads_per_particle,
                                     const unsigned int block_size,
                                     const hipDeviceProp_t& devprop);

//! Kernel driver for filling the particle types for sorting
hipError_t gpu_compute_nlist_stencil_fill_types(unsigned int* d_pids,
                                                unsigned int* d_types,
                                                const Scalar4* d_pos,
                                                const unsigned int N);

//! Wrapper to CUB sorting
void gpu_compute_nlist_stencil_sort_types(unsigned int* d_pids,
                                          unsigned int* d_pids_alt,
                                          unsigned int* d_types,
                                          unsigned int* d_types_alt,
                                          void* d_tmp_storage,
                                          size_t& tmp_storage_bytes,
                                          bool& swap,
                                          const unsigned int N);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __NEIGHBORLOSTGPUSTENCIL_CUH__
