// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairGPU.cuh"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

namespace hoomd { namespace md { namespace kernel {
template
__attribute__((visibility("default")))
hipError_t
gpu_compute_pair_forces<EvaluatorPair@_evaluator_cpp@>(const pair_args_t& pair_args,
                                   const EvaluatorPair@_evaluator_cpp@::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
