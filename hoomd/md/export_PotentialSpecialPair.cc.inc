// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialSpecialPair.h"
#include "hoomd/md/EvaluatorSpecialPair@_pair@.h"

#define EVALUATOR_CLASS EvaluatorSpecialPair@_pair@
#define EXPORT_FUNCTION export_PotentialSpecialPair@_pair@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialSpecialPair<EVALUATOR_CLASS>(m, "PotentialSpecialPair@_pair@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
