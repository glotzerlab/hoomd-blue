// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialBondGPU.cu
    \brief Defines the driver functions for computing all types of bond forces on the GPU
*/

#include "AllDriverPotentialBondGPU.cuh"
#include "EvaluatorBondFENE.h"
#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondTether.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_harmonic_forces(const bond_args_t& bond_args,
                                       const harmonic_params* d_params,
                                       unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondHarmonic>(bond_args, d_params, d_flags);
    }

hipError_t gpu_compute_fene_forces(const bond_args_t& bond_args,
                                   const fene_params* d_params,
                                   unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondFENE>(bond_args, d_params, d_flags);
    }

hipError_t gpu_compute_tether_forces(const bond_args_t& bond_args,
                                     const tether_params* d_params,
                                     unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondTether>(bond_args, d_params, d_flags);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
