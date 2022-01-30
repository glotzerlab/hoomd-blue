// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapePolyhedron.h"
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
void export_polyhedron(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapePolyhedron>(m, "IntegratorHPMCMonoPolyhedron");
    export_ComputeFreeVolume<ShapePolyhedron>(m, "ComputeFreeVolumePolyhedron");
    export_ComputeSDF<ShapePolyhedron>(m, "ComputeSDFPolyhedron");
    export_UpdaterMuVT<ShapePolyhedron>(m, "UpdaterMuVTPolyhedron");
    export_UpdaterClusters<ShapePolyhedron>(m, "UpdaterClustersPolyhedron");

    // export_ShapeMoveInterface<ShapePolyhedron>(m, "ShapeMovePolyhedron");
    export_UpdaterShape<ShapePolyhedron>(m, "UpdaterShapePolyhedron");
    export_PythonShapeMove<ShapePolyhedron>(m, "PythonShapeMovePolyhedron");
    export_ConstantShapeMove<ShapePolyhedron>(m, "ConstantShapeMovePolyhedron");

    export_ExternalFieldInterface<ShapePolyhedron>(m, "ExternalFieldPolyhedron");
    export_HarmonicField<ShapePolyhedron>(m, "ExternalFieldHarmonicPolyhedron");
    export_ExternalFieldComposite<ShapePolyhedron>(m, "ExternalFieldCompositePolyhedron");
    export_ExternalFieldWall<ShapePolyhedron>(m, "WallPolyhedron");
    export_UpdaterExternalFieldWall<ShapePolyhedron>(m, "UpdaterExternalFieldWallPolyhedron");
    export_ExternalCallback<ShapePolyhedron>(m, "ExternalCallbackPolyhedron");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapePolyhedron>(m, "IntegratorHPMCMonoPolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapePolyhedron>(m, "ComputeFreeVolumePolyhedronGPU");
    export_UpdaterClustersGPU<ShapePolyhedron>(m, "UpdaterClustersPolyhedronGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
