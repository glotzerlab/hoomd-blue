// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TableDihedralForceGPU.cuh
    \brief Declares GPU kernel code for calculating the table bond forces. Used by
   TableDihedralForceGPU.
*/

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

#ifndef __TABLEDIHEDRALFORCECOMPUTEGPU_CUH__
#define __TABLEDIHEDRALFORCECOMPUTEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes table forces on the GPU for TableDihedralForceGPU
hipError_t gpu_compute_table_dihedral_forces(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* d_pos,
                                             const BoxDim& box,
                                             const group_storage<4>* dlist,
                                             const unsigned int* dihedral_ABCD,
                                             const unsigned int pitch,
                                             const unsigned int* n_dihedrals_list,
                                             const Scalar2* d_tables,
                                             const unsigned int table_width,
                                             const Index2D& table_value,
                                             const unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
