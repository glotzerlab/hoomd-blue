// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/MeshGroupData.cuh"
#include "hoomd/ParticleData.cuh"
#include <hip/hip_runtime.h>

/*! \file MeshAreaConservationGPU.cuh
    \brief Declares GPU kernel code for calculating the area cnstraint forces. Used by
   MeshAreaConservationGPU.
*/

#ifndef __AREACONSERVATIONMESHFORCECOMPUTE_CUH__
#define __AREACONSERVATIONMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the area for MeshAreaConservationGPU
hipError_t gpu_compute_area_constraint_area(Scalar* d_sum_area,
                                            Scalar* d_sum_partial_area,
                                            const unsigned int N,
                                            const Scalar4* d_pos,
                                            const BoxDim& box,
                                            const group_storage<6>* tlist,
                                            const unsigned int* tpos_list,
                                            const Index2D tlist_idx,
                                            const unsigned int* n_triangles_list,
                                            unsigned int block_size,
                                            unsigned int num_blocks);

//! Kernel driver that computes the forces for MeshAreaConservationGPU
hipError_t gpu_compute_area_constraint_force(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* d_pos,
                                             const BoxDim& box,
                                             const Scalar area,
                                             const group_storage<6>* tlist,
                                             const unsigned int* tpos_list,
                                             const Index2D tlist_idx,
                                             const unsigned int* n_triangles_list,
                                             Scalar2* d_params,
                                             const unsigned int n_triangle_type,
                                             int block_size,
                                             unsigned int* d_flags);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
