// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairRotationalCoupling.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairRotationalCouplingGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairRotationalCoupling>(
        m,
        "AnisoPotentialPairRotationalCouplingGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
