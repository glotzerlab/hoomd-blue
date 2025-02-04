// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialPairDPDThermo.h"
#include "hoomd/md/EvaluatorPairDPDThermo@_evaluator@.h"

#define EVALUATOR_CLASS EvaluatorPairDPDThermo@_evaluator@
#define EXPORT_FUNCTION export_PotentialPairDPDThermo@_evaluator@
// clang-format on

namespace hoomd
    {
namespace md
    {

// This should not be needed as the export_ method below instantiates the template, but runtime
// errors result without these lines.
template class PotentialPair<EVALUATOR_CLASS>;
template class PotentialPairDPDThermo<EVALUATOR_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialPair<EVALUATOR_CLASS>(m, "PotentialPairConservative@_evaluator@");
    export_PotentialPairDPDThermo<EVALUATOR_CLASS>(m, "PotentialPairDPDThermo@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
