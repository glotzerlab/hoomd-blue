// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/ParticleData.cuh"

/*! \file SurfaceTensionMeshForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating the surface tension forces. Used by
   SurfaceTensionMeshForceComputeGPU.
*/

#ifndef __SURFACETENSIONMESHFORCECOMPUTE_CUH__
#define __SURFACETENSIONMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the forces for SurfaceTensionMeshForceComputeGPU
hipError_t gpu_compute_surface_tension(Scalar* d_sum_area,
                                                     Scalar* d_sum_partial_area,
                                                     const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     const BoxDim& box,
                                                     const group_storage<3>* tlist,
                                                     const Index2D tlist_idx,
                                                     const unsigned int* n_triangles_list,
                                                     unsigned int block_size,
                                                     unsigned int num_blocks);

//! Kernel driver that computes the forces for SurfaceTensionMeshForceComputeGPU
hipError_t gpu_compute_surface_tension_force(Scalar4* d_force,
                                                      Scalar* d_virial,
                                                      const size_t virial_pitch,
                                                      const unsigned int N,
                                                      const unsigned int N_tri,
                                                      const Scalar4* d_pos,
                                                      const BoxDim& box,
                                                      const group_storage<3>* tlist,
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
