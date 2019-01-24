// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


/*! \file DriverTersoffGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __DRIVER_TERSOFF_GPU_CUH__
#define __DRIVER_TERSOFF_GPU_CUH__

#include "PotentialTersoffGPU.cuh"
#include "EvaluatorTersoff.h"
#include "EvaluatorSquareDensity.h"

//! Compute Tersoff forces on the GPU with EvaluatorTersoff
cudaError_t gpu_compute_tersoff_forces(const tersoff_args_t& pair_args,
                                       const tersoff_params *d_params);

//! Compute CG vdW liquid forces on the GPU with EvaluatorSquareDensity
cudaError_t gpu_compute_sq_density_forces(const tersoff_args_t& pair_args,
                                   const Scalar2 *d_params);

#endif
