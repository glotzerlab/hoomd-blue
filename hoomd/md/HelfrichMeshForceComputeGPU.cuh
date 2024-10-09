// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file HelfrichMeshForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating the helfrich forces. Used by
   HelfrichMeshForceComputeGPU.
*/

#ifndef __HELFRICHMESHFORCECOMPUTE_CUH__
#define __HELFRICHMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the sigmas for HelfrichMeshForceComputeGPU
hipError_t gpu_compute_helfrich_sigma(Scalar* d_sigma,
                                      Scalar3* d_sigma_dash,
                                      const unsigned int N,
                                      const Scalar4* d_pos,
                                      const unsigned int* d_rtag,
                                      const BoxDim& box,
                                      const group_storage<4>* blist,
                                      const Index2D blist_idx,
                                      const unsigned int* bpos_list,
                                      const unsigned int* n_bonds_list,
                                      int block_size);

//! Kernel driver that computes the forces for HelfrichMeshForceComputeGPU
hipError_t gpu_compute_helfrich_force(Scalar4* d_force,
                                      Scalar* d_virial,
                                      const size_t virial_pitch,
                                      const unsigned int N,
                                      const Scalar4* d_pos,
                                      const unsigned int* d_rtag,
                                      const BoxDim& box,
                                      const Scalar* d_sigma,
                                      const Scalar3* d_sigma_dash,
                                      const group_storage<4>* blist,
                                      const Index2D blist_idx,
                                      const unsigned int* bpos_list,
                                      const unsigned int* n_bonds_list,
                                      Scalar* d_params,
                                      const unsigned int n_bond_type,
                                      int block_size);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
