// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AllDriverPotentialPairGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__

#include "EvaluatorPairBuckingham.h"
#include "EvaluatorPairDLVO.h"
#include "EvaluatorPairEwald.h"
#include "EvaluatorPairExpandedLJ.h"
#include "EvaluatorPairExpandedMie.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairFourier.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairLJ0804.h"
#include "EvaluatorPairLJ1208.h"
#include "EvaluatorPairLJGauss.h"
#include "EvaluatorPairMie.h"
#include "EvaluatorPairMoliere.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairOPP.h"
#include "EvaluatorPairReactionField.h"
#include "EvaluatorPairTWF.h"
#include "EvaluatorPairTable.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairZBL.h"
#include "PotentialPairDPDThermoGPU.cuh"
#include "PotentialPairGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Compute lj pair forces on the GPU with PairEvaluatorLJ
hipError_t __attribute__((visibility("default")))
gpu_compute_ljtemp_forces(const pair_args_t& pair_args,
                          const EvaluatorPairLJ::param_type* d_params);

//! Compute gauss pair forces on the GPU with PairEvaluatorGauss
hipError_t __attribute__((visibility("default")))
gpu_compute_gauss_forces(const pair_args_t& pair_args,
                         const EvaluatorPairGauss::param_type* d_params);

//! Compute expanded lj pair forces on the GPU with PairEvaluatorExpandedLJ
hipError_t __attribute__((visibility("default")))
gpu_compute_expanded_lj_forces(const pair_args_t& pair_args,
                               const EvaluatorPairExpandedLJ::param_type* d_params);

//! Compute yukawa pair forces on the GPU with PairEvaluatorGauss
hipError_t __attribute__((visibility("default")))
gpu_compute_yukawa_forces(const pair_args_t& pair_args,
                          const EvaluatorPairYukawa::param_type* d_params);

//! Compute morse pair forces on the GPU with PairEvaluatorMorse
hipError_t __attribute__((visibility("default")))
gpu_compute_morse_forces(const pair_args_t& pair_args,
                         const EvaluatorPairMorse::param_type* d_params);

//! Compute ewlad pair forces on the GPU with PairEvaluatorEwald
hipError_t __attribute__((visibility("default")))
gpu_compute_ewald_forces(const pair_args_t& pair_args,
                         const EvaluatorPairEwald::param_type* d_params);

//! Compute moliere pair forces on the GPU with EvaluatorPairMoliere
hipError_t __attribute__((visibility("default")))
gpu_compute_moliere_forces(const pair_args_t& pair_args,
                           const EvaluatorPairMoliere::param_type* d_params);

//! Compute zbl pair forces on the GPU with EvaluatorPairZBL
hipError_t __attribute__((visibility("default")))
gpu_compute_zbl_forces(const pair_args_t& pair_args, const EvaluatorPairZBL::param_type* d_params);

//! Compute force shifted lj pair forces on the GPU with PairEvaluatorForceShiftedLJ
hipError_t __attribute__((visibility("default")))
gpu_compute_force_shifted_lj_forces(const pair_args_t& args,
                                    const EvaluatorPairForceShiftedLJ::param_type* d_params);

//! Compute mie potential pair forces on the GPU with PairEvaluatorMie
hipError_t __attribute__((visibility("default")))
gpu_compute_mie_forces(const pair_args_t& args, const EvaluatorPairMie::param_type* d_params);

//! Compute expanded mie potential pair forces on the GPU with PairEvaluatorExpandedMie
hipError_t __attribute__((visibility("default")))
gpu_compute_expanded_mie_forces(const pair_args_t& args,
                                const EvaluatorPairExpandedMie::param_type* d_params);

//! Compute reaction field potential pair forces on the GPU with PairEvaluatorReactionField
hipError_t __attribute__((visibility("default")))
gpu_compute_reaction_field_forces(const pair_args_t& args,
                                  const EvaluatorPairReactionField::param_type* d_params);

//! Compute buckingham pair forces on the GPU with PairEvaluatorBuckingham
hipError_t __attribute__((visibility("default")))
gpu_compute_buckingham_forces(const pair_args_t& pair_args,
                              const EvaluatorPairBuckingham::param_type* d_params);

//! Compute lj1208 pair forces on the GPU with PairEvaluatorLJ1208
hipError_t __attribute__((visibility("default")))
gpu_compute_lj1208_forces(const pair_args_t& pair_args,
                          const EvaluatorPairLJ1208::param_type* d_params);

//! Compute lj0804 pair forces on the GPU with PairEvaluatorLJ0804
hipError_t __attribute__((visibility("default")))
gpu_compute_lj0804_forces(const pair_args_t& pair_args,
                          const EvaluatorPairLJ0804::param_type* d_params);

//! Compute DLVO potential pair forces on the GPU with EvaluatorPairDLVO
hipError_t __attribute__((visibility("default")))
gpu_compute_dlvo_forces(const pair_args_t& args, const EvaluatorPairDLVO::param_type* d_params);

//! Compute Fourier potential pair forces on the GPU with PairEvaluatorFourier
hipError_t __attribute__((visibility("default")))
gpu_compute_fourier_forces(const pair_args_t& pair_args,
                           const EvaluatorPairFourier::param_type* d_params);

//! Compute oscillating pair potential forces on the GPU with EvaluatorPairOPP
hipError_t __attribute__((visibility("default")))
gpu_compute_opp_forces(const pair_args_t& pair_args, const EvaluatorPairOPP::param_type* d_params);

//! Compute tabulated pair potential forces on the GPU with EvaluatorPairTable
hipError_t __attribute__((visibility("default")))
gpu_compute_table_forces(const pair_args_t& pair_args,
                         const EvaluatorPairTable::param_type* d_params);

//! Compute oscillating pair potential forces on the GPU with EvaluatorPairOPP
hipError_t __attribute__((visibility("default")))
gpu_compute_twf_forces(const pair_args_t& pair_args, const EvaluatorPairTWF::param_type* d_params);

//! Compute lj gauss potential pair forces on the GPU with EvaluatorLJGauss
hipError_t __attribute__((visibility("default")))
gpu_compute_lj_gauss_forces(const pair_args_t& pair_args,
                            const EvaluatorPairLJGauss::param_type* d_params);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
