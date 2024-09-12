// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoNEC.h"

#include "ComputeSDF.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"
#include "UpdaterVirtualMoveMonteCarlo.h"

#include "ShapeMoves.h"
#include "ShapeUtils.h"
#include "UpdaterShape.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterClustersGPU.h"
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
    export_UpdaterClusters<ShapeConvexPolyhedron>(m, "UpdaterClustersConvexPolyhedron");
    export_UpdaterVirtualMoveMonteCarlo<ShapeConvexPolyhedron>(m, "UpdaterVMMCConvexPolyhedron");

    export_MassProperties<ShapeConvexPolyhedron>(m, "MassPropertiesConvexPolyhedron");

    export_UpdaterShape<ShapeConvexPolyhedron>(m, "UpdaterShapeConvexPolyhedron");
    export_ShapeMoveBase<ShapeConvexPolyhedron>(m, "ShapeMoveBaseShapeConvexPolyhedron");
    export_PythonShapeMove<ShapeConvexPolyhedron>(m, "ShapeSpaceConvexPolyhedron");
    ;
    export_ElasticShapeMove<ShapeConvexPolyhedron>(m, "ElasticConvexPolyhedron");
    export_ConvexPolyhedronVertexShapeMove(m, "VertexConvexPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron>(m, "ExternalFieldConvexPolyhedron");
    export_HarmonicField<ShapeConvexPolyhedron>(m, "ExternalFieldHarmonicConvexPolyhedron");
    export_ExternalFieldWall<ShapeConvexPolyhedron>(m, "WallConvexPolyhedron");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeConvexPolyhedron>(m, "IntegratorHPMCMonoConvexPolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapeConvexPolyhedron>(m, "ComputeFreeVolumeConvexPolyhedronGPU");
    export_UpdaterClustersGPU<ShapeConvexPolyhedron>(m, "UpdaterClustersConvexPolyhedronGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
