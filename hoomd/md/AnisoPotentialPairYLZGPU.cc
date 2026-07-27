// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairYLZ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairYLZGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairYLZ>(m, "AnisoPotentialPairYLZGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
