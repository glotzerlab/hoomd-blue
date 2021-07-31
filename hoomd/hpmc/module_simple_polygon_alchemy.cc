// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
#include "ShapeSimplePolygon.h"

#include "ShapeUtils.h"
#include "ShapeMoves.h"
#include "UpdaterShape.h"


namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the shape moves used in hpmc alchemy
void export_simple_polygon_alchemy(py::module& m)
    {
    export_ShapeMoveInterface< ShapeSimplePolygon >(m, "ShapeMoveSimplePolygon");
    export_UpdaterShape< ShapeSimplePolygon >(m, "UpdaterShapeSimplePolygon");
    export_PythonShapeMove< ShapeSimplePolygon >(m, "PythonShapeMoveSimplePolygon");
    export_ConstantShapeMove< ShapeSimplePolygon >(m, "ConstantShapeMoveSimplePolygon");
    }
}
