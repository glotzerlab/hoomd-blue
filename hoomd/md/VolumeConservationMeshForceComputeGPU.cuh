// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/MeshGroupData.cuh"
#include "hoomd/ParticleData.cuh"

/*! \file MeshVolumeConservationGPU.cuh
    \brief Declares GPU kernel code for calculating the volume cnstraint forces. Used by
   MeshVolumeConservationGPU.
*/

#ifndef __MESHVOLUMECONSERVATION_CUH__
#define __MESHVOLUMECONSERVATION_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the volume for MeshVolumeConservationGPU
hipError_t gpu_compute_volume_constraint_volume(Scalar volume,
                                                const unsigned int N,
                                                const Scalar4* d_pos,
                                                const int3* d_image,
                                                const BoxDim& box,
                                                const group_storage<6>* tlist,
                                                const unsigned int* tpos_list,
                                                const Index2D tlist_idx,
                                                const unsigned int* n_triangles_list,
                                                int block_size);

//! Kernel driver that computes the forces for MeshVolumeConservationGPU
hipError_t gpu_compute_volume_constraint_force(Scalar4* d_force,
                                               Scalar* d_virial,
                                               const size_t virial_pitch,
                                               const unsigned int N,
                                               const Scalar4* d_pos,
                                               const int3* d_image,
                                               const BoxDim& box,
                                               const Scalar volume,
                                               const group_storage<6>* tlist,
                                               const unsigned int* tpos_list,
                                               const Index2D tlist_idx,
                                               const unsigned int* n_triangles_list,
                                               Scalar* d_params,
                                               const unsigned int n_triangle_type,
                                               int block_size,
                                               unsigned int* d_flags);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
