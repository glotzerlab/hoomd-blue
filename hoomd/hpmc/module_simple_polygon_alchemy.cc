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
    export_ShapeLogBoltzmann< ShapeSimplePolygon >(m, "LogBoltzmannSimplePolygon");
    export_AlchemyLogBoltzmannFunction< ShapeSimplePolygon >(m, "AlchemyLogBoltzmannSimplePolygon");
    export_UpdaterShape< ShapeSimplePolygon >(m, "UpdaterShapeSimplePolygon");
    export_PythonShapeMove< ShapeSimplePolygon >(m, "PythonShapeMoveSimplePolygon");
    export_ConstantShapeMove< ShapeSimplePolygon >(m, "ConstantShapeMoveSimplePolygon");
    }
}
