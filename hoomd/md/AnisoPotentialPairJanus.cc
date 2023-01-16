// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "PairModulator.h"
#include "GeneralEnvelope.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairMie.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<PairModulator<EvaluatorPairLJ, GeneralEnvelope>>(pybind11::module& m,
                                                                                                      const std::string& name);
template void export_AnisoPotentialPair<PairModulator<EvaluatorPairMie, GeneralEnvelope>>(pybind11::module& m,
                                                                                                       const std::string& name);

void export_AnisoPotentialPairJanusLJ(pybind11::module& m)
    {
    export_AnisoPotentialPair<PairModulator<EvaluatorPairLJ, GeneralEnvelope>>(m, "AnisoPotentialPairJanusLJ");
    }

void export_AnisoPotentialPairMie(pybind11::module& m)
    {
    export_AnisoPotentialPair<PairModulator<EvaluatorPairMie, GeneralEnvelope>>(m, "AnisoPotentialPairJanusMie");
    }


    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
