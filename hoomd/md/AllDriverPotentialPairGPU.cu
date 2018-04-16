// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander / Everyone is free to add additional potentials

/*! \file AllDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "EvaluatorPairLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairMorse.h"
#include "PotentialPairDPDThermoGPU.cuh"
#include "EvaluatorPairDPDThermo.h"
#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairEwald.h"
#include "EvaluatorPairMoliere.h"
#include "EvaluatorPairZBL.h"
#include "EvaluatorPairDPDLJThermo.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairMie.h"
#include "EvaluatorPairReactionField.h"
#include "EvaluatorPairLJ1208.h"
#include "EvaluatorPairBuckingham.h"
#include "EvaluatorPairDLVO.h"

cudaError_t gpu_compute_ljtemp_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ>(pair_args,
                                                    d_params);
    }

cudaError_t gpu_compute_gauss_forces(const pair_args_t& pair_args,
                                     const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairGauss>(pair_args,
                                                       d_params);
    }

cudaError_t gpu_compute_slj_forces(const pair_args_t& pair_args,
                                   const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairSLJ>(pair_args,
                                                     d_params);
    }

cudaError_t gpu_compute_yukawa_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairYukawa>(pair_args,
                                                        d_params);
    }


cudaError_t gpu_compute_morse_forces(const pair_args_t& pair_args,
                                      const Scalar4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairMorse>(pair_args,
                                                       d_params);
    }

cudaError_t gpu_compute_dpdthermodpd_forces(const dpd_pair_args_t& args,
                                            const Scalar2 *d_params)
    {
    return gpu_compute_dpd_forces<EvaluatorPairDPDThermo>(args,
                                                          d_params);
    }


cudaError_t gpu_compute_dpdthermo_forces(const pair_args_t& pair_args,
                                         const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairDPDThermo>(pair_args,
                                                           d_params);
    }


cudaError_t gpu_compute_ewald_forces(const pair_args_t& pair_args,
                                     const Scalar2 *d_params)
    {
    return  gpu_compute_pair_forces<EvaluatorPairEwald>(pair_args,
                                                        d_params);
    }

cudaError_t gpu_compute_moliere_forces(const pair_args_t& pair_args,
                                       const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairMoliere>(pair_args,
                                                         d_params);
    }

cudaError_t gpu_compute_zbl_forces(const pair_args_t& pair_args,
                                   const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairZBL>(pair_args,
                                                     d_params);
    }

cudaError_t gpu_compute_dpdljthermodpd_forces(const dpd_pair_args_t& args,
                                              const Scalar4 *d_params)
    {
    return gpu_compute_dpd_forces<EvaluatorPairDPDLJThermo>(args,
                                                            d_params);
    }


cudaError_t gpu_compute_dpdljthermo_forces(const pair_args_t& args,
                                           const Scalar4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairDPDLJThermo>(args,
                                                             d_params);
    }

cudaError_t gpu_compute_force_shifted_lj_forces(const pair_args_t & args,
                                                const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairForceShiftedLJ>(args,
                                                                d_params);
    }

cudaError_t gpu_compute_mie_forces(const pair_args_t & args,
                                                const Scalar4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairMie>(args,
                                                     d_params);
    }

cudaError_t gpu_compute_reaction_field_forces(const pair_args_t & args,
                                                const Scalar3 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairReactionField>(args,
                                                     d_params);
    }

cudaError_t gpu_compute_lj1208_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ1208>(pair_args,
                                                    d_params);
    }

cudaError_t gpu_compute_buckingham_forces(const pair_args_t & args,
                                                const Scalar4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairBuckingham>(args,
                                                     d_params);
    }

cudaError_t gpu_compute_dlvo_forces(const pair_args_t & args,
                                                const Scalar3 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairDLVO>(args,
                                                     d_params);
    }
