// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeSpheropolygon.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_spheropolygon_alchemy(py::module& m)
    {
    export_ShapeMoveInterface<ShapeSpheropolygon>(m, "ShapeMoveSpheropolygon");
    export_UpdaterShape<ShapeSpheropolygon>(m, "UpdaterShapeSpheropolygon");
    export_PythonShapeMove<ShapeSpheropolygon>(m, "PythonShapeMoveSpheropolygon");
    export_ConstantShapeMove<ShapeSpheropolygon>(m, "ConstantShapeMoveSpheropolygon");
    }
    } // namespace hpmc
