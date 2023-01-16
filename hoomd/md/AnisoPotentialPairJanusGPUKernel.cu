// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairLJ.h"
#include "GeneralEnvelope.h"
#include "PairModulator.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<PairModulator<EvaluatorPairLJ, GeneralEnvelope<JanusFactor>>>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairLJ::param_type* d_param,
    const EvaluatorPairLJ::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
