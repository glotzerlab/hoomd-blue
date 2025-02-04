// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BondTablePotentialGPU.cuh
    \brief Declares GPU kernel code for calculating the table bond forces. Used by
   BONDTablePotentialGPU.
*/

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

#ifndef __BONDTABLEPOTENTIALGPU_CUH__
#define __BONDTABLEPOTENTIALGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes table forces on the GPU for TablePotentialGPU
hipError_t gpu_compute_bondtable_forces(Scalar4* d_force,
                                        Scalar* d_virial,
                                        const size_t virial_pitch,
                                        const unsigned int N,
                                        const Scalar4* d_pos,
                                        const BoxDim& box,
                                        const group_storage<2>* blist,
                                        const unsigned int pitch,
                                        const unsigned int* n_bonds_list,
                                        const unsigned int n_bond_type,
                                        const Scalar2* d_tables,
                                        const Scalar4* d_params,
                                        const unsigned int table_width,
                                        const Index2D& table_value,
                                        unsigned int* d_flags,
                                        const unsigned int block_size,
                                        const hipDeviceProp_t& devprop);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
