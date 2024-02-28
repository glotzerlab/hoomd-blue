// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairALJ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairALJ3D(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairALJ<3>>(m, "AnisoPotentialPairALJ3D");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
