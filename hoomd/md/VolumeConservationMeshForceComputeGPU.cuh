// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"
#include <hip/hip_runtime.h>

/*! \file MeshVolumeConservationGPU.cuh
    \brief Declares GPU kernel code for calculating the volume constraint forces. Used by
   MeshVolumeConservationGPU.
*/

#ifndef __VOLUMECONSERVATIONMESHFORCECOMPUTE_CUH__
#define __VOLUMECONSERVATIONMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the volume for MeshVolumeConservationGPU
hipError_t gpu_compute_volume_constraint_volume(Scalar* d_sum_volume,
                                                Scalar* d_sum_partial_volume,
                                                const unsigned int N,
                                                const unsigned int tN,
                                                const Scalar4* d_pos,
                                                const int3* d_image,
                                                const BoxDim& box,
                                                const group_storage<3>* tlist,
                                                const unsigned int* tpos_list,
                                                const Index2D tlist_idx,
						const bool ignore_type,
                                                const unsigned int* n_triangles_list,
                                                unsigned int block_size,
                                                unsigned int num_blocks);

//! Kernel driver that computes the forces for MeshVolumeConservationGPU
hipError_t gpu_compute_volume_constraint_force(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const unsigned int* gN,
                                               const unsigned int aN,
                                               const Scalar4* d_pos,
                                               const int3* d_image,
                                               const BoxDim& box,
                                               const Scalar* volume,
                                               const group_storage<3>* tlist,
                                               const unsigned int* tpos_list,
                                               const Index2D tlist_idx,
                                               const unsigned int* n_triangles_list,
                                               Scalar2* d_params,
                                               const bool ignore_type,
                                               int block_size);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
