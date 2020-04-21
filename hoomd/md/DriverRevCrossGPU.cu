// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

/*! \file DriverRevCrossGPU.cu
    \brief Defines the driver functions for computing all types of three-body forces on the GPU
*/

#include "DriverRevCrossGPU.cuh"
#include "EvaluatorRevCross.h"

hipError_t gpu_compute_revcross_forces(const revcross_args_t& pair_args,
                                       const revcross_params *d_params)
    {
    return gpu_compute_tripletrevcross_forces<EvaluatorRevCross>(pair_args,
                                                           d_params);
    }
