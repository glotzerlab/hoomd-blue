// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file DriverTersoffGPU.cu
    \brief Defines the driver functions for computing all types of three-body forces on the GPU
*/

#include "DriverTersoffGPU.cuh"
#include "EvaluatorRevCross.h"
#include "EvaluatorSquareDensity.h"
#include "EvaluatorTersoff.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_tersoff_forces(const tersoff_args_t& pair_args,
                                      const EvaluatorTersoff::param_type* d_params)
    {
    return gpu_compute_triplet_forces<EvaluatorTersoff>(pair_args, d_params);
    }

hipError_t gpu_compute_sq_density_forces(const tersoff_args_t& pair_args,
                                         const EvaluatorSquareDensity::param_type* d_params)
    {
    return gpu_compute_triplet_forces<EvaluatorSquareDensity>(pair_args, d_params);
    }

hipError_t gpu_compute_revcross_forces(const tersoff_args_t& pair_args,
                                       const EvaluatorRevCross::param_type* d_params)
    {
    return gpu_compute_triplet_forces<EvaluatorRevCross>(pair_args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
