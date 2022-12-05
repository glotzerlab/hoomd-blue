// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairLJ.h"
#include "GeneralEnvelope.h"
#include "JanusFactor.h"
#include "PairModulator.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairJanusLJGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<PairModulator<EvaluatorPairLJ, GeneralEnvelope<JanusFactor>>>(
        m,
        "AnisoPotentialPairJanusLJGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
