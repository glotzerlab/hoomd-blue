// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LJ0804DriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "EvaluatorPairLJ0804.h"
#include "AllDriverPotentialPairGPU.cuh"

hipError_t gpu_compute_lj0804_forces(const pair_args_t& pair_args,
                                      const EvaluatorPairLJ0804::param_type *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ0804>(pair_args,
                                                    d_params);
    }

