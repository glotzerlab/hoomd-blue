// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/PotentialMeshTriangleGPU.h"
#include "hoomd/md/EvaluatorPair@_mesh_triangle_pair@.h"

#define EVALUATOR_CLASS EvaluatorPair@_mesh_triangle_pair@
#define EXPORT_FUNCTION export_PotentialMeshTriangle@_mesh_triangle_pair@GPU
// clang-format on

namespace hoomd
    {
namespace md
    {

// Use CPU class from another compilation unit to reduce compile time and compiler memory usage.
extern template class PotentialMeshTriangle<EVALUATOR_CLASS>;

namespace detail
    {

void EXPORT_FUNCTION(pybind11::module& m)
    {
    export_PotentialMeshTriangleGPU<EVALUATOR_CLASS>(
        m,
        "PotentialMeshTriangle@_mesh_triangle_pair@GPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
