// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EvaluatorExternalElectricField.h"
#include "EvaluatorExternalPeriodic.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairMie.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorWalls.h"
#include "PotentialExternalGPU.cuh"
#include "WallData.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// Instantiate external evaluator templates
//! Evaluator for External Periodic potentials.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorExternalPeriodic>(const external_potential_args_t& external_potential_args,
                                    const typename EvaluatorExternalPeriodic::param_type* d_params,
                                    const typename EvaluatorExternalPeriodic::field_type* d_field);
//! Evaluator for electric fields
template hipError_t __attribute__((visibility("default"))) gpu_cpef<EvaluatorExternalElectricField>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorExternalElectricField::param_type* d_params,
    const typename EvaluatorExternalElectricField::field_type* d_field);
//! Evaluator for Lennard-Jones pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairLJ>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairLJ>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairLJ>::field_type* d_field);
//! Evaluator for Gaussian pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairGauss>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairGauss>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairGauss>::field_type* d_field);
//! Evaluator for Yukawa pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairYukawa>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairYukawa>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairYukawa>::field_type* d_field);
//! Evaluator for Morse pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairMorse>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairMorse>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairMorse>::field_type* d_field);
//! Evaluator for Force Shifted Lennard-Jones pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairForceShiftedLJ>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::field_type* d_field);
//! Evaluator for Mie pair potential.
template hipError_t __attribute__((visibility("default")))
gpu_cpef<EvaluatorWalls<EvaluatorPairMie>>(
    const external_potential_args_t& external_potential_args,
    const typename EvaluatorWalls<EvaluatorPairMie>::param_type* d_params,
    const typename EvaluatorWalls<EvaluatorPairMie>::field_type* d_field);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
