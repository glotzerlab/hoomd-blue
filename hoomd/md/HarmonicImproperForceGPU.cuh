// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BondedGroupData.cuh"

/*! \file HarmonicImproperForceGPU.cuh
    \brief Declares GPU kernel code for calculating the harmonic improper forces. Used by HarmonicImproperForceComputeGPU.
*/

#ifndef __HARMONICIMPROPERFORCEGPU_CUH__
#define __HARMONICIMPROPERFORCEGPU_CUH__

//! Kernel driver that computes harmonic IMPROPER forces for HarmonicImproperForceComputeGPU
cudaError_t gpu_compute_harmonic_improper_forces(Scalar4* d_force,
                                                 Scalar* d_virial,
                                                 const unsigned int virial_pitch,
                                                 const unsigned int N,
                                                 const Scalar4 *d_pos,
                                                 const BoxDim& box,
                                                 const group_storage<4> *tlist,
                                                 const unsigned int *dihedral_ABCD,
                                                 const unsigned int pitch,
                                                 const unsigned int *n_dihedrals_list,
                                                 Scalar2 *d_params,
                                                 unsigned int n_improper_types,
                                                 int block_size);

#endif
