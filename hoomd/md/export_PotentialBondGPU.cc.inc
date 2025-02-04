// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialBondGPU.h"
#include "hoomd/md/EvaluatorBond@_bond@.h"

#define EVALUATOR_CLASS EvaluatorBond@_bond@
#define EXPORT_FUNCTION export_PotentialBond@_bond@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class PotentialBond<EVALUATOR_CLASS, BondData>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialBondGPU<EVALUATOR_CLASS>(m, "PotentialBond@_bond@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
