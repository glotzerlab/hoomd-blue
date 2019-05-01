// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialSpecialPairGPU.cuh
    \brief Declares driver functions for computing all types of special pair forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_SPECIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_SPECIAL_PAIR_GPU_CUH__

// Internally, the GPU kernels for bonds are used
#include "PotentialBondGPU.cuh"

//! Compute LJ bond forces on the GPU with BondEvaluatorLJ
cudaError_t gpu_compute_lj_forces(const bond_args_t& bond_args,
                                    const Scalar3 *d_params,
                                    unsigned int *d_flags);

//! Compute Coulomb bond forces on the GPU with BondEvaluatorCoulomb
cudaError_t gpu_compute_coulomb_forces(const bond_args_t& bond_args,
                                    const Scalar2 *d_params,
                                    unsigned int *d_flags);
#endif
