// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialTersoffGPU.cuh"
#include "hoomd/md/Evaluator@_evaluator@.h"

namespace hoomd { namespace md { namespace kernel {
template
__attribute__((visibility("default")))
hipError_t
gpu_compute_triplet_forces<Evaluator@_evaluator@>(const tersoff_args_t& pair_args,
                                      const typename Evaluator@_evaluator@::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
