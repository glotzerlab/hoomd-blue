// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairGPU.cuh"
#include "EvaluatorPairExample.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template __attribute__((visibility("default"))) hipError_t
gpu_compute_pair_forces<EvaluatorPairExample>(const pair_args_t& pair_args,
                                         const EvaluatorPairExample::param_type* d_params);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
