// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeSimplePolygon.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_simple_polygon_alchemy(py::module& m)
    {
    export_ShapeMoveInterface<ShapeSimplePolygon>(m, "ShapeMoveSimplePolygon");
    export_UpdaterShape<ShapeSimplePolygon>(m, "UpdaterShapeSimplePolygon");
    export_PythonShapeMove<ShapeSimplePolygon>(m, "PythonShapeMoveSimplePolygon");
    export_ConstantShapeMove<ShapeSimplePolygon>(m, "ConstantShapeMoveSimplePolygon");
    }
    } // namespace hpmc
