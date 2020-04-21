// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

/*! \file DriverRevCrossGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __DRIVER_REVCROSS_GPU_CUH__
#define __DRIVER_REVCROSS_GPU_CUH__

#include "PotentialRevCrossGPU.cuh"
#include "EvaluatorRevCross.h"

//! Compute RevCross forces on the GPU with EvaluatorRevCross
hipError_t gpu_compute_revcross_forces(const revcross_args_t& pair_args,
                                       const revcross_params *d_params);

#endif
