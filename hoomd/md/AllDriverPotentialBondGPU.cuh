// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialBondGPU.cuh
    \brief Declares driver functions for computing all types of bond forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_BOND_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_BOND_GPU_CUH__

#include "PotentialBondGPU.cuh"

//! Compute harmonic bond forces on the GPU with BondEvaluatorHarmonic
hipError_t __attribute__((visibility("default")))
gpu_compute_harmonic_forces(const bond_args_t& bond_args,
                            const Scalar2 *d_params,
                            unsigned int *d_flags);

//! Compute FENE bond forces on the GPU with BondEvaluatorFENE
hipError_t __attribute__((visibility("default")))
gpu_compute_fene_forces(const bond_args_t& bond_args,
                        const Scalar4 *d_params,
                        unsigned int *d_flags);
#endif
