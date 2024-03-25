// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairGB.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairGB>(pybind11::module& m,
                                                         const std::string& name);

void export_AnisoPotentialPairGB(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairGB>(m, "AnisoPotentialPairGB");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
