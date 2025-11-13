// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EvaluatorPairFrictionLJBase.h"
#include "EvaluatorPairFrictionLJVariants.h"
#include "FrictionPairGPU.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_FrictionPairFrictionLJLinearGPU(pybind11::module& m)
    {
    export_FrictionPairGPU<EvaluatorPairFrictionLJLinear>(m, "FrictionPairFrictionLJLinearGPU");
    }

void export_FrictionPairFrictionLJCoulombGPU(pybind11::module& m)
    {
    export_FrictionPairGPU<EvaluatorPairFrictionLJCoulomb>(m, "FrictionPairFrictionLJCoulombGPU");
    }

void export_FrictionPairFrictionLJCoulombNewtonGPU(pybind11::module& m)
    {
    export_FrictionPairGPU<EvaluatorPairFrictionLJCoulombNewton>(
        m,
        "FrictionPairFrictionLJCoulombNewtonGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
