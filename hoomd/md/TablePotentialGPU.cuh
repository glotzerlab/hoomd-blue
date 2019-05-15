// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TablePotentialGPU.cuh
    \brief Declares GPU kernel code for calculating the table pair forces. Used by TablePotentialGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"

#ifndef __TABLEPOTENTIALGPU_CUH__
#define __TABLEPOTENTIALGPU_CUH__

//! Kernel driver that computes table forces on the GPU for TablePotentialGPU
cudaError_t gpu_compute_table_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const unsigned int n_ghost,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const unsigned int *d_head_list,
                                     const Scalar2 *d_tables,
                                     const Scalar4 *d_params,
                                     const unsigned int size_nlist,
                                     const unsigned int ntypes,
                                     const unsigned int table_width,
                                     const unsigned int block_size,
                                     const GPUPartition& gpu_partition);

#endif
