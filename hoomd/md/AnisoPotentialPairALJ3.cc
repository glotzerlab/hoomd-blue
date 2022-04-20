// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairALJ3.h"

namespace hoomd
    {
namespace md
    {

template class EvaluatorPairALJ<3>;
template class AnisoPotentialPair<EvaluatorPairALJ<3>>;

namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairALJ<3>>(pybind11::module& m,
                                                             const std::string& name);

void export_AnisoPotentialPairALJ3D(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairALJ<3>>(m, "AnisoPotentialPairALJ3D");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
