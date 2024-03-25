// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPair.h"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

#define EVALUATOR_CLASS EvaluatorPair@_evaluator_cpp@
#define EXPORT_FUNCTION export_PotentialPair@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialPair<EVALUATOR_CLASS>(m, "PotentialPair@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
