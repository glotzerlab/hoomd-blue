// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialTersoffGPU.cuh"
#include "hoomd/md/Evaluator@_evaluator@.h"

#define EVALUATOR_CLASS Evaluator@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_triplet_forces<EVALUATOR_CLASS>(const tersoff_args_t& pair_args,
                                            const typename EVALUATOR_CLASS::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
