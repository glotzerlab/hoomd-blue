// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairDPDThermoGPU.h"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorPairDPDThermo@_evaluator@
#define EXPORT_FUNCTION export_PotentialPairDPDThermo@_evaluator@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {

// Use CPU classes from another compilation unit to reduce compile time and compiler memory usage.
extern template class PotentialPair<EVALUATOR_CLASS>;
extern template class PotentialPairDPDThermo<EVALUATOR_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialPairDPDThermoGPU<EVALUATOR_CLASS>(m, "PotentialPairDPDThermo@_evaluator@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
