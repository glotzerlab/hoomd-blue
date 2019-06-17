// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file AllDriverAnisoPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of anisotropic pair forces on the GPU
*/

#ifndef __ALL_DRIVER_ANISO_POTENTIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_ANISO_POTENTIAL_PAIR_GPU_CUH__

#include "AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairGB.h"
#include "EvaluatorPairDipole.h"
#include "EvaluatorPair2DALJ.h"
#include "EvaluatorPairALJTable.h"

//! Compute dipole forces and torques on the GPU with EvaluatorPairDipole

cudaError_t gpu_compute_pair_aniso_forces_gb(const a_pair_args_t&,
            const EvaluatorPairGB::param_type*);

cudaError_t gpu_compute_pair_aniso_forces_dipole(const a_pair_args_t&,
            const EvaluatorPairDipole::param_type*);

cudaError_t gpu_compute_pair_aniso_forces_2DALJ(const a_pair_args_t&,
            const EvaluatorPair2DALJ::param_type*);        

cudaError_t gpu_compute_pair_aniso_forces_ALJTable(const a_pair_args_t&,
            const EvaluatorPairALJTable::param_type*);   

#endif
