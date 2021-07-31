// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
#include "ShapeUnion.h"

#include "ShapeUtils.h"
#include "ShapeMoves.h"
#include "UpdaterShape.h"


namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the shape moves used in hpmc alchemy
void export_union_sphere_alchemy(py::module& m)
    {
    export_ShapeMoveInterface< ShapeUnion<ShapeSphere> >(m, "ShapeMoveSphereUnion");
    export_UpdaterShape< ShapeUnion<ShapeSphere> >(m, "UpdaterShapeSphereUnion");
    export_PythonShapeMove< ShapeUnion<ShapeSphere> >(m, "PythonShapeMoveSphereUnion");
    export_ConstantShapeMove< ShapeUnion<ShapeSphere> >(m, "ConstantShapeMoveSphereUnion");
    }
}
