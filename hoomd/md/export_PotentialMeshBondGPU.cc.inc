// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialBondGPU.h"
#include "hoomd/md/EvaluatorBond@_bond@.h"

#define EVALUATOR_CLASS EvaluatorBond@_bond@
#define EXPORT_FUNCTION export_PotentialMeshBond@_bond@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class PotentialBond<EVALUATOR_CLASS, MeshBondData>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialMeshBondGPU<EVALUATOR_CLASS>(m, "PotentialMeshBond@_bond@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
