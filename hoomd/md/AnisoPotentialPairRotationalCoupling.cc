// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairRotationalCoupling.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairRotationalCoupling>(pybind11::module& m,
                                                                         const std::string& name);

void export_AnisoPotentialPairRotationalCoupling(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairRotationalCoupling>(
        m,
        "AnisoPotentialPairRotationalCoupling");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
