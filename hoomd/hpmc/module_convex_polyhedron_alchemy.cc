// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeConvexPolyhedron.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_convex_polyhedron_alchemy(py::module& m)
    {
    export_MassProperties<ShapeConvexPolyhedron>(m, "MassPropertiesConvexPolyhedron");
    export_ShapeMoveInterface<ShapeConvexPolyhedron>(m, "ShapeMoveConvexPolyhedron");
    export_ElasticShapeMove<ShapeConvexPolyhedron>(m, "ElasticShapeMoveConvexPolyhedron");
    export_ConvexPolyhedronGeneralizedShapeMove<ShapeConvexPolyhedron>(
        m,
        "GeneralizedShapeMoveConvexPolyhedron");
    export_UpdaterShape<ShapeConvexPolyhedron>(m, "UpdaterShapeConvexPolyhedron");
    export_PythonShapeMove<ShapeConvexPolyhedron>(m, "PythonShapeMoveConvexPolyhedron");
    export_ConstantShapeMove<ShapeConvexPolyhedron>(m, "ConstantShapeMoveConvexPolyhedron");
    }
    } // namespace hpmc
