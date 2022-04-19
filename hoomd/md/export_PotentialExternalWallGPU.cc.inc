// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialExternalGPU.h"
#include "hoomd/md/EvaluatorPair@_evaluator@.h"
#include "hoomd/md/EvaluatorWalls.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialExternalWall@_evaluator@GPU(pybind11::module &m)
    {
    export_PotentialExternalGPU<EvaluatorWalls<EvaluatorPair@_evaluator@>>(m, "WallsPotential@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
