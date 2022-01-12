// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file DriverTersoffGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU
*/

#ifndef __DRIVER_TERSOFF_GPU_CUH__
#define __DRIVER_TERSOFF_GPU_CUH__

#include "EvaluatorRevCross.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorTersoff.h"
#include "PotentialTersoffGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Compute Tersoff forces on the GPU with EvaluatorTersoff
hipError_t gpu_compute_tersoff_forces(const tersoff_args_t& pair_args,
                                      const EvaluatorTersoff::param_type* d_params);

//! Compute CG vdW liquid forces on the GPU with EvaluatorSquareDensity
hipError_t gpu_compute_sq_density_forces(const tersoff_args_t& pair_args,
                                         const EvaluatorSquareDensity::param_type* d_params);

//! Compute RevCross forces on the GPU with EvaluatorRevCross
hipError_t gpu_compute_revcross_forces(const tersoff_args_t& pair_args,
                                       const EvaluatorRevCross::param_type* d_params);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
