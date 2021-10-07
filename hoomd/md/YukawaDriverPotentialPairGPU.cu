// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file YukawaDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairYukawa.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_yukawa_forces(const pair_args_t& pair_args,
                                     const EvaluatorPairYukawa::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairYukawa>(pair_args, d_params);
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
