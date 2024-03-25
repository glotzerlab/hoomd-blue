// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file OPLSDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the OPLS dihedral forces. Used by
   OPLSDihedralForceComputeGPU.
*/

#ifndef __OPLSDIHEDRALFORCEGPU_CUH__
#define __OPLSDIHEDRALFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes OPLS dihedral forces for OPLSDihedralForceComputeGPU
hipError_t gpu_compute_opls_dihedral_forces(Scalar4* d_force,
                                            Scalar* d_virial,
                                            const size_t virial_pitch,
                                            const unsigned int N,
                                            const Scalar4* d_pos,
                                            const BoxDim& box,
                                            const group_storage<4>* tlist,
                                            const unsigned int* dihedral_ABCD,
                                            const unsigned int pitch,
                                            const unsigned int* n_dihedrals_list,
                                            const Scalar4* d_params,
                                            const unsigned int n_dihedral_types,
                                            const int block_size,
                                            const int warp_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
