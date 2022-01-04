// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ReactionFieldDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairReactionField.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_reaction_field_forces(const pair_args_t& args,
                                             const EvaluatorPairReactionField::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairReactionField>(args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
