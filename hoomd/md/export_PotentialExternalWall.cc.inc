// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialExternal.h"
#include "hoomd/md/EvaluatorPair@_evaluator@.h"
#include "hoomd/md/EvaluatorWalls.h"

#define EVALUATOR_CLASS EvaluatorWalls<EvaluatorPair@_evaluator@>
#define EXPORT_FUNCTION export_PotentialExternalWall@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialExternal<EVALUATOR_CLASS>(m, "WallsPotential@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
