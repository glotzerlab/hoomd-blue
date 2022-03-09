// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"

#include "ShapeMoves.h"
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
void export_convex_spheropolyhedron(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSpheropolyhedron>(m, "IntegratorHPMCMonoSpheropolyhedron");
    export_ComputeFreeVolume<ShapeSpheropolyhedron>(m, "ComputeFreeVolumeSpheropolyhedron");
    export_ComputeSDF<ShapeSpheropolyhedron>(m, "ComputeSDFConvexSpheropolyhedron");
    export_UpdaterMuVT<ShapeSpheropolyhedron>(m, "UpdaterMuVTConvexSpheropolyhedron");
    export_UpdaterClusters<ShapeSpheropolyhedron>(m, "UpdaterClustersConvexSpheropolyhedron");

    export_UpdaterShape<ShapeSpheropolyhedron>(m, "UpdaterShapeSpheropolyhedron");
    export_ShapeMoveBase<ShapeSpheropolyhedron>(m, "ShapeMoveBaseShapeSpheropolyhedron");
    export_PythonShapeMove<ShapeSpheropolyhedron>(m, "PythonShapeMoveSpheropolyhedron");
    export_ConstantShapeMove<ShapeSpheropolyhedron>(m, "ConstantShapeMoveSpheropolyhedron");

    export_ExternalFieldInterface<ShapeSpheropolyhedron>(m, "ExternalFieldSpheropolyhedron");
    export_HarmonicField<ShapeSpheropolyhedron>(m, "ExternalFieldHarmonicSpheropolyhedron");
    export_ExternalFieldComposite<ShapeSpheropolyhedron>(m,
                                                         "ExternalFieldCompositeSpheropolyhedron");
    export_ExternalFieldWall<ShapeSpheropolyhedron>(m, "WallConvexSpheropolyhedron");
    export_ExternalCallback<ShapeSpheropolyhedron>(m, "ExternalCallbackSpheropolyhedron");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeSpheropolyhedron>(m, "IntegratorHPMCMonoSpheropolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapeSpheropolyhedron>(m, "ComputeFreeVolumeSpheropolyhedronGPU");
    export_UpdaterClustersGPU<ShapeSpheropolyhedron>(m, "UpdaterClustersConvexSpheropolyhedronGPU");

#endif
    }
    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
