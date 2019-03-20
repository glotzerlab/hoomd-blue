// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


/*! \file DriverTersoffGPU.cu
    \brief Defines the driver functions for computing all types of three-body forces on the GPU
*/

#include "DriverTersoffGPU.cuh"
#include "EvaluatorTersoff.h"

cudaError_t gpu_compute_tersoff_forces(const tersoff_args_t& pair_args,
                                       const tersoff_params *d_params)
    {
    return gpu_compute_triplet_forces<EvaluatorTersoff>(pair_args,
                                                        d_params);
    }

cudaError_t gpu_compute_sq_density_forces(const tersoff_args_t& pair_args,
                                   const Scalar2 *d_params)
    {
    return gpu_compute_triplet_forces<EvaluatorSquareDensity>(pair_args,
                                                            d_params);
    }
