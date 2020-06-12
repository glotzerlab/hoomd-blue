#include "ShapeConvexPolyhedron.h"

#include "ShapeUtils.h"
#include "ShapeMoves.h"
#include "UpdaterShape.h"


namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the shape moves used in hpmc alchemy
void export_convex_polyhedron_alchemy(py::module& m)
    {
    export_MassProperties< ShapeConvexPolyhedron >(m, "MassPropertiesConvexPolyhedron");
    export_ShapeMoveInterface< ShapeConvexPolyhedron >(m, "ShapeMoveConvexPolyhedron");
    export_ShapeLogBoltzmann< ShapeConvexPolyhedron >(m, "LogBoltzmannConvexPolyhedron");
    export_ElasticShapeMove< ShapeConvexPolyhedron >(m, "ElasticShapeMoveConvexPolyhedron");
    export_ShapeSpringLogBoltzmannFunction<ShapeConvexPolyhedron >(m, "ShapeSpringLogBoltzmannConvexPolyhedron");
    export_AlchemyLogBoltzmannFunction< ShapeConvexPolyhedron >(m, "AlchemyLogBoltzmannConvexPolyhedron");
    export_ConvexPolyhedronGeneralizedShapeMove< ShapeConvexPolyhedron >(m, "GeneralizedShapeMoveConvexPolyhedron");
    export_UpdaterShape< ShapeConvexPolyhedron >(m, "UpdaterShapeConvexPolyhedron");
    export_PythonShapeMove< ShapeConvexPolyhedron >(m, "PythonShapeMoveConvexPolyhedron");
    export_ConstantShapeMove< ShapeConvexPolyhedron >(m, "ConstantShapeMoveConvexPolyhedron");
    }
}
