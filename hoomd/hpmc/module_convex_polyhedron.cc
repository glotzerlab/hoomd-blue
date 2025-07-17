// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoNEC.h"

#include "ComputeSDF.h"
#include "ShapeConvexPolyhedron.h"

#include "ExternalFieldWall.h"

#include "UpdaterGCA.h"
#include "UpdaterMuVT.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterGCAGPU.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Export the base HPMCMono integrators
void export_convex_polyhedron(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeConvexPolyhedron>(m, "IntegratorHPMCMonoConvexPolyhedron");
    export_IntegratorHPMCMonoNEC<ShapeConvexPolyhedron>(m, "IntegratorHPMCMonoNECConvexPolyhedron");
    export_ComputeFreeVolume<ShapeConvexPolyhedron>(m, "ComputeFreeVolumeConvexPolyhedron");
    export_ComputeSDF<ShapeConvexPolyhedron>(m, "ComputeSDFConvexPolyhedron");
    export_UpdaterMuVT<ShapeConvexPolyhedron>(m, "UpdaterMuVTConvexPolyhedron");
    export_UpdaterGCA<ShapeConvexPolyhedron>(m, "UpdaterGCAConvexPolyhedron");

    export_MassProperties<ShapeConvexPolyhedron>(m, "MassPropertiesConvexPolyhedron");

    export_UpdaterShape<ShapeConvexPolyhedron>(m, "UpdaterShapeConvexPolyhedron");
    export_ShapeMoveBase<ShapeConvexPolyhedron>(m, "ShapeMoveBaseShapeConvexPolyhedron");
    export_PythonShapeMove<ShapeConvexPolyhedron>(m, "ShapeSpaceConvexPolyhedron");
    ;
    export_ElasticShapeMove<ShapeConvexPolyhedron>(m, "ElasticConvexPolyhedron");
    export_ConvexPolyhedronVertexShapeMove(m, "VertexConvexPolyhedron");

    export_ExternalFieldWall<ShapeConvexPolyhedron>(m, "WallConvexPolyhedron");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeConvexPolyhedron>(m, "IntegratorHPMCMonoConvexPolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapeConvexPolyhedron>(m, "ComputeFreeVolumeConvexPolyhedronGPU");
    export_UpdaterGCAGPU<ShapeConvexPolyhedron>(m, "UpdaterGCAConvexPolyhedronGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
