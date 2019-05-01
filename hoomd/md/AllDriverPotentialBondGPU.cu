// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialBondGPU.cu
    \brief Defines the driver functions for computing all types of bond forces on the GPU
*/

#include "EvaluatorBondHarmonic.h"
#include "EvaluatorBondFENE.h"
#include "AllDriverPotentialBondGPU.cuh"

cudaError_t gpu_compute_harmonic_forces(const bond_args_t& bond_args,
                                        const Scalar2 *d_params,
                                        unsigned int *d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondHarmonic>(bond_args,
                                                    d_params,
                                                    d_flags);
    }

cudaError_t gpu_compute_fene_forces(const bond_args_t& bond_args,
                                   const Scalar4 *d_params,
                                   unsigned int *d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondFENE>(bond_args,
                                                     d_params,
                                                     d_flags);
    }
