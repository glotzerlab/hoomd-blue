// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairDipole.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairDipole>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairDipole::param_type* d_param,
    const EvaluatorPairDipole::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
