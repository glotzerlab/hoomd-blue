// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairALJ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairALJ2DGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairALJ<2>>(m, "AnisoPotentialPairALJ2DGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
