// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
    return gpu_compute_bond_forces<EvaluatorBondHarmonic, 2>(bond_args, d_params, d_flags);
    }

hipError_t gpu_compute_fene_forces(const bond_args_t& bond_args,
                                   const fene_params* d_params,
                                   unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondFENE, 2>(bond_args, d_params, d_flags);
    }

hipError_t gpu_compute_tether_forces(const bond_args_t& bond_args,
                                     const tether_params* d_params,
                                     unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondTether, 2>(bond_args, d_params, d_flags);
    }

hipError_t gpu_compute_harmonic_forces(const meshbond_args_t& meshbond_args,
                                       const harmonic_params* d_params,
                                       unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondHarmonic, 4>(meshbond_args, d_params, d_flags);
    }

hipError_t gpu_compute_fene_forces(const meshbond_args_t& meshbond_args,
                                   const fene_params* d_params,
                                   unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondFENE, 4>(meshbond_args, d_params, d_flags);
    }

hipError_t gpu_compute_tether_forces(const meshbond_args_t& meshbond_args,
                                     const tether_params* d_params,
                                     unsigned int* d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondTether, 4>(meshbond_args, d_params, d_flags);
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
