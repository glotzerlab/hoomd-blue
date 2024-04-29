// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairGPU.h"
#include "hoomd/md/EvaluatorPair@_evaluator_cpp@.h"

#define EVALUATOR_CLASS EvaluatorPair@_evaluator_cpp@
#define EXPORT_FUNCTION export_PotentialPair@_evaluator@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class PotentialPair<EVALUATOR_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialPairGPU<EVALUATOR_CLASS>(m, "PotentialPair@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
