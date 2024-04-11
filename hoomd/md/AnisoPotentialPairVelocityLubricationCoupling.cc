// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairVelocityLubricationCoupling.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairVelocityLubricationCoupling>(pybind11::module& m,
                                                                          const std::string& name);

void export_AnisoPotentialPairVelocityLubricationCoupling(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairVelocityLubricationCoupling>(
        m,
        "AnisoPotentialPairVelocityLubricationCoupling");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
