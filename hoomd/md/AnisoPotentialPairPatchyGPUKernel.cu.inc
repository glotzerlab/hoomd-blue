// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.cuh"
#include "hoomd/md/PatchEnvelope.h"
#include "hoomd/md/PairModulator.h"

// clang-format off
#include "hoomd/md/EvaluatorPair@_evaluator@.h"
#define EVALUATOR_CLASS EvaluatorPair@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
    template hipError_t __attribute__((visibility("default")))
    gpu_compute_pair_aniso_forces<PairModulator<EVALUATOR_CLASS, PatchEnvelope>>(
        const a_pair_args_t& pair_args,
        const PairModulator<EVALUATOR_CLASS, PatchEnvelope>::param_type* d_param,
        const PairModulator<EVALUATOR_CLASS, PatchEnvelope>::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
