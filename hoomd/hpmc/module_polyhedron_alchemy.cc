// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapePolyhedron.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_polyhedron_alchemy(py::module& m)
    {
    export_ShapeMoveInterface<ShapePolyhedron>(m, "ShapeMovePolyhedron");
    export_UpdaterShape<ShapePolyhedron>(m, "UpdaterShapePolyhedron");
    export_PythonShapeMove<ShapePolyhedron>(m, "PythonShapeMovePolyhedron");
    export_ConstantShapeMove<ShapePolyhedron>(m, "ConstantShapeMovePolyhedron");
    }
    } // namespace hpmc
