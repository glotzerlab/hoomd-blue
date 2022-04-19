// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairALJ2.h"

namespace hoomd { namespace md {

template class EvaluatorPairALJ<2>;
template class AnisoPotentialPair<EvaluatorPairALJ<2>>;

namespace detail {

template void export_AnisoPotentialPair<EvaluatorPairALJ<2>>(pybind11::module& m, const std::string& name);

void export_AnisoPotentialPairALJ2D(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairALJ<2>>(m, "AnisoPotentialPairALJ2D");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
