// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file TriangleAreaConservationMeshForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating the area conservation forces. Used by
   TriangleAreaConservationMeshForceComputeGPU.
*/

#ifndef __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_CUH__
#define __TRIANGLEAREACONSERVATIONMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the forces for TriangleAreaConservationMeshForceComputeGPU
hipError_t gpu_compute_TriangleAreaConservation_force(Scalar4* d_force,
                                                      Scalar* d_virial,
                                                      const size_t virial_pitch,
                                                      const unsigned int N,
                                                      const Scalar4* d_pos,
                                                      const BoxDim& box,
                                                      const group_storage<3>* tlist,
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
