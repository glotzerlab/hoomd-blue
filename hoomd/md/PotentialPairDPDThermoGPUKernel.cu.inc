// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialPairDPDThermoGPU.cuh"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

namespace hoomd { namespace md { namespace kernel {
template
__attribute__((visibility("default")))
hipError_t
gpu_compute_dpd_forces<EvaluatorPairDPDThermo@_evaluator@>(const dpd_pair_args_t& args,
                                  const typename EvaluatorPairDPDThermo@_evaluator@::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
