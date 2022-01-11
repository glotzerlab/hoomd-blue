// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AllDriverPotentialSpecialPairGPU.cu
    \brief Defines the driver functions for computing all types of special pair forces on the GPU

*/

#include "AllDriverPotentialSpecialPairGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! LJ special pair potential, internal
hipError_t gpu_compute_lj_forces(const bond_args_t& bond_args,
                                 const special_lj_params* d_params,
                                 unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorSpecialPairLJ>(bond_args, d_params, d_flags);
    }

//! Coulomb special pair potential, internal
hipError_t gpu_compute_coulomb_forces(const bond_args_t& bond_args,
                                      const special_coulomb_params* d_params,
                                      unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorSpecialPairCoulomb>(bond_args, d_params, d_flags);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
