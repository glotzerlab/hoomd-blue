// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __CELLLISTGPU_CUH__
#define __CELLLISTGPU_CUH__

#if defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include "GPUPartition.cuh"
#include "HOOMDMath.h"
#include "Index1D.h"
#include "ParticleData.cuh"

/*! \file CellListGPU.cuh
    \brief Declares GPU kernel code for cell list generation on the GPU
*/

namespace hoomd
    {
//! Kernel driver for gpu_compute_cell_list_kernel()
void gpu_compute_cell_list(unsigned int* d_cell_size,
                           Scalar4* d_xyzf,
                           uint2* d_type_body,
                           Scalar4* d_cell_orientation,
                           unsigned int* d_cell_idx,
                           uint3* d_conditions,
                           const Scalar4* d_pos,
                           const Scalar4* d_orientation,
                           const Scalar* d_charge,
                           const Scalar* d_diameter,
                           const unsigned int* d_body,
                           const unsigned int N,
                           const unsigned int n_ghost,
                           const unsigned int Nmax,
                           const bool flag_charge,
                           const bool flag_type,
                           const BoxDim& box,
                           const Index3D& ci,
                           const Index2D& cli,
                           const Scalar3& ghost_width,
                           const unsigned int block_size,
                           const GPUPartition& gpu_partition);

//! Driver function to combine the cell lists from different GPUs into one
hipError_t gpu_combine_cell_lists(const unsigned int* d_cell_size_scratch,
                                  unsigned int* d_cell_size,
                                  const unsigned int* d_idx_scratch,
                                  unsigned int* d_idx,
                                  const Scalar4* d_xyzf_scratch,
                                  Scalar4* d_xyzf,
                                  const uint2* d_type_body_scratch,
                                  uint2* d_type_body,
                                  const Scalar4* d_cell_orientation_scratch,
                                  Scalar4* d_cell_orientation,
                                  const Index2D cli,
                                  unsigned int ngpu,
                                  const unsigned int block_size,
                                  const unsigned int Nmax,
                                  uint3* d_conditions,
                                  const GPUPartition& gpu_partition);

hipError_t gpu_sort_cell_list(unsigned int* d_cell_size,
                              Scalar4* d_xyzf,
                              Scalar4* d_xyzf_new,
                              uint2* d_type_body,
                              uint2* d_type_body_new,
                              Scalar4* d_cell_orientation,
                              Scalar4* d_cell_orientation_new,
                              unsigned int* d_cell_idx,
                              unsigned int* d_cell_idx_new,
                              uint2* d_sort_idx,
                              unsigned int* d_sort_permutation,
                              const Index3D ci,
                              const Index2D cli);

    } // end namespace hoomd
#endif
