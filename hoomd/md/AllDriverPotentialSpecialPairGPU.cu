// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser / Anyone is free to add their own pair potentials here

/*! \file AllDriverPotentialSpecialPairGPU.cu
    \brief Defines the driver functions for computing all types of special pair forces on the GPU

*/

#include "EvaluatorSpecialPairLJ.h"
#include "EvaluatorSpecialPairCoulomb.h"
#include "AllDriverPotentialSpecialPairGPU.cuh"

//! LJ special pair potential, internal
cudaError_t gpu_compute_lj_forces(const bond_args_t& bond_args,
                                   const Scalar3 *d_params,
                                   unsigned int *d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorSpecialPairLJ>(bond_args,
                                                     d_params,
                                                     d_flags);
    }

//! Coulomb special pair potential, internal
cudaError_t gpu_compute_coulomb_forces(const bond_args_t& bond_args,
                                   const Scalar2 *d_params,
                                   unsigned int *d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorSpecialPairCoulomb>(bond_args,
                                                     d_params,
                                                     d_flags);
    }
