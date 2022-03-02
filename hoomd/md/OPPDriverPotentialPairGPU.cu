// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file OPPDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing opp pair forces on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairOPP.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_opp_forces(const pair_args_t& pair_args,
                                  const EvaluatorPairOPP::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairOPP>(pair_args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
