// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "PairModulator.h"
#include "SingleEnvelope.h"
#include "EvaluatorPairLJ.h"
#include "EvaluatorPairMie.h"
#include "EvaluatorPairYukawa.h"
#include "EvaluatorPairExpandedGaussian.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {



// template void export_AnisoPotentialPair<PairModulator<EvaluatorPairLJ, SingleEnvelope>>(pybind11::module& m, const std::string& name);
// template void export_AnisoPotentialPair<PairModulator<EvaluatorPairMie, SingleEnvelope>>(pybind11::module& m, const std::string& name);
// template void export_AnisoPotentialPair<PairModulator<EvaluatorPairYukawa, SingleEnvelope>>(pybind11::module& m, const std::string& name);
// template void export_AnisoPotentialPair<PairModulator<EvaluatorPairExpandedGaussian, SingleEnvelope>>(pybind11::module& m, const std::string& name);


void export_AnisoPotentialPairJanusSingleLJ(pybind11::module& m)
    {
    export_AnisoPotentialPair<PairModulator<EvaluatorPairLJ, SingleEnvelope>>(m, "AnisoPotentialPairJanusSingleLJ");
    }

void export_AnisoPotentialPairJanusSingleMie(pybind11::module& m)
    {
    export_AnisoPotentialPair<PairModulator<EvaluatorPairMie, SingleEnvelope>>(m, "AnisoPotentialPairJanusSingleMie");
    }

void export_AnisoPotentialPairJanusSingleYukawa(pybind11::module& m)
    {
    export_AnisoPotentialPair<PairModulator<EvaluatorPairYukawa, SingleEnvelope>>(m, "AnisoPotentialPairJanusSingleYukawa");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
