// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialExternal.h"
#include "hoomd/md/EvaluatorPair@_evaluator@.h"
#include "hoomd/md/EvaluatorWalls.h"

namespace hoomd { namespace md { namespace detail {

void export_PotentialExternalWall@_evaluator@(pybind11::module &m)
    {
    export_PotentialExternal<EvaluatorWalls<EvaluatorPair@_evaluator@>>(m, "WallsPotential@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
