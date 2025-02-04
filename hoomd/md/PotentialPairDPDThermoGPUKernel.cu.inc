// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairDPDThermoGPU.cuh"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorPairDPDThermo@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_dpd_forces<EVALUATOR_CLASS>(const dpd_pair_args_t& args,
                                        const typename EVALUATOR_CLASS::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
