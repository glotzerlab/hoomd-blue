// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file HarmonicDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the harmonic dihedral forces. Used by
   HarmonicDihedralForceComputeGPU.
*/

#ifndef __HARMONICDIHEDRALFORCEGPU_CUH__
#define __HARMONICDIHEDRALFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes harmonic dihedral forces for HarmonicDihedralForceComputeGPU
hipError_t gpu_compute_harmonic_dihedral_forces(Scalar4* d_force,
                                                Scalar* d_virial,
                                                const size_t virial_pitch,
                                                const unsigned int N,
                                                const Scalar4* d_pos,
                                                const BoxDim& box,
                                                const group_storage<4>* tlist,
                                                const unsigned int* dihedral_ABCD,
                                                const unsigned int pitch,
                                                const unsigned int* n_dihedrals_list,
                                                Scalar4* d_params,
                                                unsigned int n_dihedral_types,
                                                int block_size,
                                                int warp_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
