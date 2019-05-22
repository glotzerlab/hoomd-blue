// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: phillicl

/*! \file TableAngleForceGPU.cuh
    \brief Declares GPU kernel code for calculating the table bond forces. Used by TableAngleForceGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

#ifndef __TABLEANGLEFORCECOMPUTEGPU_CUH__
#define __TABLEANGLEFORCECOMPUTEGPU_CUH__

//! Kernel driver that computes table forces on the GPU for TableAngleForceGPU
cudaError_t gpu_compute_table_angle_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim &box,
                                     const group_storage<3> *alist,
                                     const unsigned int *apos_list,
                                     const unsigned int pitch,
                                     const unsigned int *n_angles_list,
                                     const Scalar2 *d_tables,
                                     const unsigned int table_width,
                                     const Index2D &table_value,
                                     const unsigned int block_size);

#endif
