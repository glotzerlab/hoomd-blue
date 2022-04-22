// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialExternalGPU.cuh"
#include "hoomd/md/EvaluatorExternal@_evaluator@.h"

namespace hoomd { namespace md { namespace kernel {
template
__attribute__((visibility("default")))
hipError_t
gpu_compute_potential_external_forces<EvaluatorExternal@_evaluator@>(const external_potential_args_t& external_potential_args,
                                    const typename EvaluatorExternal@_evaluator@::param_type* d_params,
                                    const typename EvaluatorExternal@_evaluator@::field_type* d_field);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
