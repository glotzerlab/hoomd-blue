// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeConvexPolygon.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterExternalFieldWall.h"
#include "UpdaterMuVT.h"

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
void export_convex_polygon(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeConvexPolygon>(m, "IntegratorHPMCMonoConvexPolygon");
    export_ComputeFreeVolume<ShapeConvexPolygon>(m, "ComputeFreeVolumeConvexPolygon");
    export_ComputeSDF<ShapeConvexPolygon>(m, "ComputeSDFConvexPolygon");
    export_UpdaterMuVT<ShapeConvexPolygon>(m, "UpdaterMuVTConvexPolygon");
    export_UpdaterClusters<ShapeConvexPolygon>(m, "UpdaterClustersConvexPolygon");

    export_ExternalFieldInterface<ShapeConvexPolygon>(m, "ExternalFieldConvexPolygon");
    export_LatticeField<ShapeConvexPolygon>(m, "ExternalFieldLatticeConvexPolygon");
    export_ExternalFieldComposite<ShapeConvexPolygon>(m, "ExternalFieldCompositeConvexPolygon");
    // export_ExternalFieldWall<ShapeConvexPolygon>(m, "WallConvexPolygon");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>(m,
    // "UpdaterExternalFieldWallConvexPolygon");
    export_ExternalCallback<ShapeConvexPolygon>(m, "ExternalCallbackConvexPolygon");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeConvexPolygon>(m, "IntegratorHPMCMonoConvexPolygonGPU");
    export_ComputeFreeVolumeGPU<ShapeConvexPolygon>(m, "ComputeFreeVolumeConvexPolygonGPU");
    export_UpdaterClustersGPU<ShapeConvexPolygon>(m, "UpdaterClustersConvexPolygonGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
