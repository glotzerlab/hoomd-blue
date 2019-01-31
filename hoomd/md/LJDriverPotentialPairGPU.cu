// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LJDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "EvaluatorPairLJ.h"
#include "AllDriverPotentialPairGPU.cuh"
cudaError_t gpu_compute_ljtemp_forces(const pair_args_t& pair_args,
                                      const Scalar2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ>(pair_args,
                                                    d_params);
    }


