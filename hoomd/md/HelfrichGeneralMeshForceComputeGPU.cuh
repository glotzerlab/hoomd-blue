// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichMeshParameters.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

/*! \file HelfrichGeneralMeshForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating the helfrich forces. Used by
   HelfrichGeneralMeshForceComputeGPU.
*/

#ifndef __HELFRICHGENERALMESHFORCECOMPUTE_CUH__
#define __HELFRICHGENERALMESHFORCECOMPUTE_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes the sigmas for HelfrichMeshForceComputeGPU
hipError_t gpu_compute_generalhelfrich_sigma(Scalar* d_sigma,
                                      Scalar3* d_sigma_dash,
                                      Scalar4* d_normal,
                                      const unsigned int N,
                                      const Scalar4* d_pos,
                                      const unsigned int* d_rtag,
                                      const BoxDim& box,
                                      const group_storage<4>* blist,
                                      const Index2D blist_idx,
                                      const unsigned int* bpos_list,
                                      const unsigned int* n_bonds_list,
                                      int block_size);

//! Kernel driver that computes the forces for HelfrichGeneralMeshForceComputeGPU
hipError_t gpu_compute_generalhelfrich_force(Scalar4* d_force,
                                      Scalar* d_virial,
                                      const size_t virial_pitch,
                                      const unsigned int N,
                                      const Scalar4* d_pos,
                                      const unsigned int* d_rtag,
                                      const BoxDim& box,
                                      const Scalar* d_sigma,
                                      const Scalar3* d_sigma_dash,
                                      const Scalar4* d_normal,
                                      const group_storage<4>* blist,
                                      const Index2D blist_idx,
                                      const unsigned int* bpos_list,
                                      const unsigned int* n_bonds_list,
                                      helfrich_param_t* d_params,
                                      const unsigned int n_bond_type,
                                      int block_size);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
