// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python

#include "ShapeConvexPolygon.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_convex_polygon_alchemy(py::module& m)
    {
    export_ShapeMoveInterface<ShapeConvexPolygon>(m, "ShapeMoveConvexPolygon");
    export_UpdaterShape<ShapeConvexPolygon>(m, "UpdaterShapeConvexPolygon");
    export_PythonShapeMove<ShapeConvexPolygon>(m, "PythonShapeMoveConvexPolygon");
    export_ConstantShapeMove<ShapeConvexPolygon>(m, "ConstantShapeMoveConvexPolygon");
    }

    } // namespace hpmc
