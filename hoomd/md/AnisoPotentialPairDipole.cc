// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairDipole.h"

namespace hoomd { namespace md {
template class AnisoPotentialPair<EvaluatorPairDipole>;

namespace detail {
template void export_AnisoPotentialPair<EvaluatorPairDipole>(pybind11::module& m, const std::string& name);
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
