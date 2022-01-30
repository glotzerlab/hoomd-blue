// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterExternalFieldWall.h"
#include "UpdaterMuVT.h"

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
    export_ComputeFreeVolume<ShapeConvexPolyhedron>(m, "ComputeFreeVolumeConvexPolyhedron");
    export_ComputeSDF<ShapeConvexPolyhedron>(m, "ComputeSDFConvexPolyhedron");
    export_UpdaterMuVT<ShapeConvexPolyhedron>(m, "UpdaterMuVTConvexPolyhedron");
    export_UpdaterClusters<ShapeConvexPolyhedron>(m, "UpdaterClustersConvexPolyhedron");

    export_MassProperties<ShapeConvexPolyhedron>(m, "MassPropertiesConvexPolyhedron");
    // export_ShapeMoveInterface<ShapeConvexPolyhedron>(m, "ShapeMoveConvexPolyhedron");
    export_ElasticShapeMove<ShapeConvexPolyhedron>(m, "ElasticShapeMoveConvexPolyhedron");
    export_ConvexPolyhedronVertexShapeMove(m, "VertexShapeMoveConvexPolyhedron");
    // export_ConvexPolyhedronGeneralizedShapeMove<ShapeConvexPolyhedron>(
    //     m,
    //     "GeneralizedShapeMoveConvexPolyhedron");
    export_UpdaterShape<ShapeConvexPolyhedron>(m, "UpdaterShapeConvexPolyhedron");
    export_PythonShapeMove<ShapeConvexPolyhedron>(m, "PythonShapeMoveConvexPolyhedron");
    export_ConstantShapeMove<ShapeConvexPolyhedron>(m, "ConstantShapeMoveConvexPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron>(m, "ExternalFieldConvexPolyhedron");
    export_HarmonicField<ShapeConvexPolyhedron>(m, "ExternalFieldHarmonicConvexPolyhedron");
    export_ExternalFieldComposite<ShapeConvexPolyhedron>(m,
                                                         "ExternalFieldCompositeConvexPolyhedron");
    export_ExternalFieldWall<ShapeConvexPolyhedron>(m, "WallConvexPolyhedron");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron>(
        m,
        "UpdaterExternalFieldWallConvexPolyhedron");
    export_ExternalCallback<ShapeConvexPolyhedron>(m, "ExternalCallbackConvexPolyhedron");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeConvexPolyhedron>(m, "IntegratorHPMCMonoConvexPolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapeConvexPolyhedron>(m, "ComputeFreeVolumeConvexPolyhedronGPU");
    export_UpdaterClustersGPU<ShapeConvexPolyhedron>(m, "UpdaterClustersConvexPolyhedronGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
