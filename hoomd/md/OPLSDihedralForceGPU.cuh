// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ksil

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BondedGroupData.cuh"

/*! \file OPLSDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the OPLS dihedral forces. Used by OPLSDihedralForceComputeGPU.
*/

#ifndef __OPLSDIHEDRALFORCEGPU_CUH__
#define __OPLSDIHEDRALFORCEGPU_CUH__

//! Kernel driver that computes OPLS dihedral forces for OPLSDihedralForceComputeGPU
cudaError_t gpu_compute_opls_dihedral_forces(Scalar4* d_force,
                                                Scalar* d_virial,
                                                const unsigned int virial_pitch,
                                                const unsigned int N,
                                                const Scalar4 *d_pos,
                                                const BoxDim& box,
                                                const group_storage<4> *tlist,
                                                const unsigned int *dihedral_ABCD,
                                                const unsigned int pitch,
                                                const unsigned int *n_dihedrals_list,
                                                const Scalar4 *d_params,
                                                const unsigned int n_dihedral_types,
                                                const int block_size);

#endif
