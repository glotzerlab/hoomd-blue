// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: jglaser / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialSpecialPairGPU.cuh
    \brief Declares driver functions for computing all types of special pair forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_SPECIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_SPECIAL_PAIR_GPU_CUH__

// Internally, the GPU kernels for bonds are used
#include "EvaluatorSpecialPairCoulomb.h"
#include "EvaluatorSpecialPairLJ.h"
#include "PotentialBondGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Compute LJ bond forces on the GPU with BondEvaluatorLJ
hipError_t __attribute__((visibility("default")))
gpu_compute_lj_forces(const bond_args_t& bond_args,
                      const special_lj_params* d_params,
                      unsigned int* d_flags);

//! Compute Coulomb bond forces on the GPU with BondEvaluatorCoulomb
hipError_t __attribute__((visibility("default")))
gpu_compute_coulomb_forces(const bond_args_t& bond_args,
                           const special_coulomb_params* d_params,
                           unsigned int* d_flags);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
