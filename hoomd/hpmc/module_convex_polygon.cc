// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"

#include "ShapeConvexPolygon.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalCallback.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterClusters.h"

#ifdef ENABLE_HIP
#include "IntegratorHPMCMonoGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif




namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_convex_polygon(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolygon >(m, "IntegratorHPMCMonoConvexPolygon");
    export_ComputeFreeVolume< ShapeConvexPolygon >(m, "ComputeFreeVolumeConvexPolygon");
    export_AnalyzerSDF< ShapeConvexPolygon >(m, "AnalyzerSDFConvexPolygon");
    export_UpdaterMuVT< ShapeConvexPolygon >(m, "UpdaterMuVTConvexPolygon");
    export_UpdaterClusters< ShapeConvexPolygon >(m, "UpdaterClustersConvexPolygon");

    export_ExternalFieldInterface<ShapeConvexPolygon>(m, "ExternalFieldConvexPolygon");
    export_LatticeField<ShapeConvexPolygon>(m, "ExternalFieldLatticeConvexPolygon");
    export_ExternalFieldComposite<ShapeConvexPolygon>(m, "ExternalFieldCompositeConvexPolygon");
    export_RemoveDriftUpdater<ShapeConvexPolygon>(m, "RemoveDriftUpdaterConvexPolygon");
    // export_ExternalFieldWall<ShapeConvexPolygon>(m, "WallConvexPolygon");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>(m, "UpdaterExternalFieldWallConvexPolygon");
    export_ExternalCallback<ShapeConvexPolygon>(m, "ExternalCallbackConvexPolygon");

    #ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU< ShapeConvexPolygon >(m, "IntegratorHPMCMonoGPUConvexPolygon");
    export_ComputeFreeVolumeGPU< ShapeConvexPolygon >(m, "ComputeFreeVolumeGPUConvexPolygon");
    #endif
    }

}
