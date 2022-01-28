// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeSpheropolyhedron.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {

//! Export the shape moves used in hpmc alchemy
void export_convex_spheropolyhedron_alchemy(py::module& m)
    {
    export_ShapeMoveInterface<ShapeSpheropolyhedron>(m, "ShapeMoveSpheropolyhedron");
    export_UpdaterShape<ShapeSpheropolyhedron>(m, "UpdaterShapeSpheropolyhedron");
    export_PythonShapeMove<ShapeSpheropolyhedron>(m, "PythonShapeMoveSpheropolyhedron");
    export_ConstantShapeMove<ShapeSpheropolyhedron>(m, "ConstantShapeMoveSpheropolyhedron");
    }
    } // namespace hpmc
