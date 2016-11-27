// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#include "WallData.h"
#include "PotentialExternalGPU.cuh"
#include "EvaluatorWalls.h"
#include "EvaluatorExternalPeriodic.h"
#include "EvaluatorExternalElectricField.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairGauss.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairSLJ.h"
#include "EvaluatorPairMorse.h"
#include "EvaluatorPairForceShiftedLJ.h"
#include "EvaluatorPairMie.h"


template< class evaluator >
cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                     const typename evaluator::param_type *d_params,
                     const typename evaluator::field_type *d_field)
    {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_external_forces_kernel<evaluator>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(external_potential_args.block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( external_potential_args.N / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);
        unsigned int bytes = (sizeof(typename evaluator::field_type)/sizeof(int)+1)*sizeof(int);

        // run the kernel
        gpu_compute_external_forces_kernel<evaluator><<<grid, threads, bytes>>>(external_potential_args.d_force,
                                                                                external_potential_args.d_virial,
                                                                                external_potential_args.virial_pitch,
                                                                                external_potential_args.N,
                                                                                external_potential_args.d_pos,
                                                                                external_potential_args.d_diameter,
                                                                                external_potential_args.d_charge,
                                                                                external_potential_args.box,
                                                                                d_params,
                                                                                d_field);

        return cudaSuccess;
    };

//Instantiate external evaluator templates

//! Evaluator for External Periodic potentials.
template cudaError_t gpu_cpef<EvaluatorExternalPeriodic>(const external_potential_args_t& external_potential_args, const typename EvaluatorExternalPeriodic::param_type *d_params, const typename EvaluatorExternalPeriodic::field_type *d_field);
//! Evaluator for electric fields
template cudaError_t gpu_cpef<EvaluatorExternalElectricField>(const external_potential_args_t& external_potential_args, const typename EvaluatorExternalElectricField::param_type *d_params, const typename EvaluatorExternalElectricField::field_type *d_field);
//! Evaluator for Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairLJ>::field_type *d_field);
//! Evaluator for Gaussian pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairGauss> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairGauss>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairGauss>::field_type *d_field);
//! Evaluator for Yukawa pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairYukawa> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairYukawa>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairYukawa>::field_type *d_field);
//! Evaluator for Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairSLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairSLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairSLJ>::field_type *d_field);
//! Evaluator for Morse pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairMorse> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairMorse>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairMorse>::field_type *d_field);
//! Evaluator for Force Shifted Lennard-Jones pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairForceShiftedLJ> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairForceShiftedLJ>::field_type *d_field);
//! Evaluator for Mie pair potential.
template cudaError_t gpu_cpef<EvaluatorWalls<EvaluatorPairMie> >(const external_potential_args_t& external_potential_args, const typename EvaluatorWalls<EvaluatorPairMie>::param_type *d_params, const typename EvaluatorWalls<EvaluatorPairMie>::field_type *d_field);
