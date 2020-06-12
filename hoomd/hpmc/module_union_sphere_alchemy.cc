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
    export_ShapeLogBoltzmann< ShapeUnion<ShapeSphere> >(m, "LogBoltzmannSphereUnion");
    export_AlchemyLogBoltzmannFunction< ShapeUnion<ShapeSphere> >(m, "AlchemyLogBoltzmannSphereUnion");
    export_UpdaterShape< ShapeUnion<ShapeSphere> >(m, "UpdaterShapeSphereUnion");
    export_PythonShapeMove< ShapeUnion<ShapeSphere> >(m, "PythonShapeMoveSphereUnion");
    export_ConstantShapeMove< ShapeUnion<ShapeSphere> >(m, "ConstantShapeMoveSphereUnion");
    }
}
