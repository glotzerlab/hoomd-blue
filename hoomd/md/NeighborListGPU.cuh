// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __NEIGHBORLISTGPU_CUH__
#define __NEIGHBORLISTGPU_CUH__

/*! \file NeighborListGPU.cuh
    \brief Declares GPU kernel code for cell list generation on the GPU
*/

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/GPUPartition.cuh"

//! Kernel driver for gpu_nlist_needs_update_check_new_kernel()
cudaError_t gpu_nlist_needs_update_check_new(unsigned int *d_result,
                                             const Scalar4 *d_last_pos,
                                             const Scalar4 *d_pos,
                                             const unsigned int N,
                                             const BoxDim& box,
                                             const Scalar *d_rcut_max,
                                             const Scalar r_buff,
                                             const unsigned int ntypes,
                                             const Scalar lambda_min,
                                             const Scalar3 lambda,
                                             const unsigned int checkn,
                                             const GPUPartition& gpu_partition);

//! Kernel driver for gpu_nlist_filter_kernel()
cudaError_t gpu_nlist_filter(unsigned int *d_n_neigh,
                             unsigned int *d_nlist,
                             const unsigned int *d_head_list,
                             const unsigned int *d_n_ex,
                             const unsigned int *d_ex_list,
                             const Index2D& exli,
                             const unsigned int N,
                             const unsigned int block_size);

//! Kernel driver to build head list on gpu
cudaError_t gpu_nlist_build_head_list(unsigned int *d_head_list,
                                      unsigned int *d_req_size_nlist,
                                      const unsigned int *d_Nmax,
                                      const Scalar4 *d_pos,
                                      const unsigned int N,
                                      const unsigned int n_types,
                                      const unsigned int block_size);


//! GPU function to update the exclusion list on the device
cudaError_t gpu_update_exclusion_list(const unsigned int *d_tag,
                                const unsigned int *d_rtag,
                                const unsigned int *d_n_ex_tag,
                                const unsigned int *d_ex_list_tag,
                                const Index2D& ex_list_tag_indexer,
                                unsigned int *d_n_ex_idx,
                                unsigned int *d_ex_list_idx,
                                const Index2D& ex_list_indexer,
                                const unsigned int N);

#endif
