// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

/*! \file CGCMMForceGPU.cuh
    \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by CGCMMForceComputeGPU.
*/

#ifndef __CGCMMFORCEGPU_CUH__
#define __CGCMMFORCEGPU_CUH__

//! Kernel driver that computes lj forces on the GPU for CGCMMForceComputeGPU
cudaError_t gpu_compute_cgcmm_forces(Scalar4* d_force,
                                     Scalar* d_virial,
                                     const unsigned int virial_pitch,
                                     const unsigned int N,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const unsigned int *d_head_list,
                                     const Scalar4 *d_coeffs,
                                     const unsigned int size_nlist,
                                     const unsigned int coeff_width,
                                     const Scalar r_cutsq,
                                     const unsigned int block_size);

#endif
