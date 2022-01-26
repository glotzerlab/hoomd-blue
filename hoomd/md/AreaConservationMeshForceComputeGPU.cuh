// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/MeshGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file AreaConservationMeshForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating the area conservation forces. Used by
   AreaConservationMeshForceComputeGPU.
*/

#ifndef __AREACONSERVATIONMESHFORCECOMPUTE_CUH__
#define __AREACONSERVATIONMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// //! Kernel driver that computes the surface area for AreaConservationMeshForceComputeGPU
// hipError_t gpu_compute_areaconservation_area(Scalar* area,
//                                              const unsigned int N,
//                                              const Scalar4* d_pos,
//                                              const BoxDim& box,
//                                              const group_storage<6>* tlist,
//                                              const Index2D tlist_idx,
//                                              const unsigned int* n_triangles_list,
//                                              int block_size);


//! Kernel driver that computes the forces for AreaConservationMeshForceComputeGPU
hipError_t gpu_compute_areaconservation_force(Scalar* d_area,
                                              Scalar4* d_force,
                                              Scalar* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const Scalar4* d_pos,
                                              const BoxDim& box,
                                              const group_storage<6>* tlist,
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
