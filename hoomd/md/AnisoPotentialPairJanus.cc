// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairJanusLJ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairJanusLJ>(pybind11::module& m,
                                                              const std::string& name);
template void export_AnisoPotentialPair<EvaluatorPairJanusMie(pybind11::module& m,
                                                              const std::string& name)

void export_AnisoPotentialPairJanusLJ(pybind11::module& m)
    {
    auto JanusEnvelope = GeneralModulator<ModulatorJanus>;
    export_AnisoPotentialPair<PairModulator<EvaluatorPairLJ, JanusEnvelope>>(m, "PotentialPairJanusLJ");
    }

void export_AnisoPotentialPairMie(pybind11::module& m)
    {
    auto JanusEnvelope = GeneralModulator<ModulatorJanus>;
    export_AnisoPotentialPair<PairModulator<EvaluatorPairMie, JanusEnvelope>>(m, "PotentialPairJanusMie");
    }


    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
