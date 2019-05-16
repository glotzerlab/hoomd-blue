// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Everyone is free to add additional potentials

/*! \file AllDriverPotentialPairGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_PAIR_GPU_CUH__

#include "PotentialPairGPU.cuh"
#include "PotentialPairDPDThermoGPU.cuh"
#include "EvaluatorPairDPDThermo.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "EvaluatorPairFourier.h"

//! Compute lj pair forces on the GPU with PairEvaluatorLJ
cudaError_t gpu_compute_ljtemp_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params);

//! Compute gauss pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_gauss_forces(const pair_args_t& pair_args,
                                     const Scalar2 *d_params);

//! Compute slj pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_slj_forces(const pair_args_t& pair_args,
                                   const Scalar2 *d_params);

//! Compute yukawa pair forces on the GPU with PairEvaluatorGauss
cudaError_t gpu_compute_yukawa_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params);

//! Compute morse pair forces on the GPU with PairEvaluatorMorse
cudaError_t gpu_compute_morse_forces(const pair_args_t& pair_args,
                                      const Scalar4 *d_params);

//! Compute dpd thermostat on GPU with PairEvaluatorDPDThermo
cudaError_t gpu_compute_dpdthermodpd_forces(const dpd_pair_args_t& args,
                                            const Scalar2 *d_params);

//! Compute dpd conservative force on GPU with PairEvaluatorDPDThermo
cudaError_t gpu_compute_dpdthermo_forces(const pair_args_t& pair_args,
                                         const Scalar2 *d_params);

//! Compute ewlad pair forces on the GPU with PairEvaluatorEwald
cudaError_t gpu_compute_ewald_forces(const pair_args_t& pair_args,
                                     const Scalar2 *d_params);

//! Compute moliere pair forces on the GPU with EvaluatorPairMoliere
cudaError_t gpu_compute_moliere_forces(const pair_args_t& pair_args,
                                       const Scalar2 *d_params);

//! Compute zbl pair forces on the GPU with EvaluatorPairZBL
cudaError_t gpu_compute_zbl_forces(const pair_args_t& pair_args,
                                   const Scalar2 *d_params);

//! Compute dpdlj thermostat on GPU with PairEvaluatorDPDThermo
cudaError_t gpu_compute_dpdljthermodpd_forces(const dpd_pair_args_t& args,
                                              const Scalar4 *d_params);

//! Compute dpdlj conservative force on GPU with PairEvaluatorDPDThermo
cudaError_t gpu_compute_dpdljthermo_forces(const pair_args_t& args,
                                           const Scalar4 *d_params);

//! Compute force shifted lj pair forces on the GPU with PairEvaluatorForceShiftedLJ
cudaError_t gpu_compute_force_shifted_lj_forces(const pair_args_t & args,
                                                const Scalar2 *d_params);

//! Compute mie potential pair forces on the GPU with PairEvaluatorMie
cudaError_t gpu_compute_mie_forces(const pair_args_t & args,
                                                const Scalar4 *d_params);

//! Compute mie potential pair forces on the GPU with PairEvaluatorReactionField
cudaError_t gpu_compute_reaction_field_forces(const pair_args_t & args,
                                                const Scalar3 *d_params);

//! Compute buckingham pair forces on the GPU with PairEvaluatorBuckingham
cudaError_t gpu_compute_buckingham_forces(const pair_args_t& pair_args,
                                      const Scalar4 *d_params);

//! Compute lj1208 pair forces on the GPU with PairEvaluatorLJ1208
cudaError_t gpu_compute_lj1208_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params);

//! Compute DLVO potential pair forces on the GPU with EvaluatorPairDLVO
cudaError_t gpu_compute_dlvo_forces(const pair_args_t & args,
                                                const Scalar3 *d_params);

//! Compute Fourier potential pair forces on the GPU with PairEvaluatorFourier
cudaError_t gpu_compute_fourier_forces(const pair_args_t & pair_args,
                                            const typename EvaluatorPairFourier::param_type *d_params);

#endif
