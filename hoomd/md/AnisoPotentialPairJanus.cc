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

void export_AnisoPotentialPairJanusLJ(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairJanus>(m, "AnisoPotentialPairGB");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
