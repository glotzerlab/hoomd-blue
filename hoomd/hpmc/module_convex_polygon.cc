// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "IntegratorHPMCMonoImplicitNew.h"
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
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "IntegratorHPMCMonoImplicitNewGPU.h"
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
    #ifdef ENABLE_HPMC_REINSERT
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolygon >(m, "IntegratorHPMCMonoImplicitConvexPolygon");
    #endif
    export_IntegratorHPMCMonoImplicitNew< ShapeConvexPolygon >(m, "IntegratorHPMCMonoImplicitNewConvexPolygon");
    export_ComputeFreeVolume< ShapeConvexPolygon >(m, "ComputeFreeVolumeConvexPolygon");
    export_AnalyzerSDF< ShapeConvexPolygon >(m, "AnalyzerSDFConvexPolygon");
    export_UpdaterMuVT< ShapeConvexPolygon >(m, "UpdaterMuVTConvexPolygon");
    #ifdef ENABLE_HPMC_REINSERT
    export_UpdaterMuVTImplicit< ShapeConvexPolygon, IntegratorHPMCMonoImplicit<ShapeConvexPolygon> >(m, "UpdaterMuVTImplicitConvexPolygon");
    #endif
    export_UpdaterMuVTImplicit< ShapeConvexPolygon, IntegratorHPMCMonoImplicitNew<ShapeConvexPolygon> >(m, "UpdaterMuVTImplicitNewConvexPolygon");
    export_UpdaterClusters< ShapeConvexPolygon >(m, "UpdaterClustersConvexPolygon");
    #ifdef ENABLE_HPMC_REINSERT
    export_UpdaterClustersImplicit< ShapeConvexPolygon, IntegratorHPMCMonoImplicit<ShapeConvexPolygon> >(m, "UpdaterClustersImplicitConvexPolygon");
    #endif
    export_UpdaterClustersImplicit< ShapeConvexPolygon, IntegratorHPMCMonoImplicitNew<ShapeConvexPolygon> >(m, "UpdaterClustersImplicitNewConvexPolygon");

    export_ExternalFieldInterface<ShapeConvexPolygon>(m, "ExternalFieldConvexPolygon");
    export_LatticeField<ShapeConvexPolygon>(m, "ExternalFieldLatticeConvexPolygon");
    export_ExternalFieldComposite<ShapeConvexPolygon>(m, "ExternalFieldCompositeConvexPolygon");
    export_RemoveDriftUpdater<ShapeConvexPolygon>(m, "RemoveDriftUpdaterConvexPolygon");
    // export_ExternalFieldWall<ShapeConvexPolygon>(m, "WallConvexPolygon");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>(m, "UpdaterExternalFieldWallConvexPolygon");
    export_ExternalCallback<ShapeConvexPolygon>(m, "ExternalCallbackConvexPolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeConvexPolygon >(m, "IntegratorHPMCMonoGPUConvexPolygon");
    #ifdef ENABLE_HPMC_REINSERT
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolygon >(m, "IntegratorHPMCMonoImplicitGPUConvexPolygon");
    #endif
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeConvexPolygon >(m, "IntegratorHPMCMonoImplicitNewGPUConvexPolygon");
    export_ComputeFreeVolumeGPU< ShapeConvexPolygon >(m, "ComputeFreeVolumeGPUConvexPolygon");
    #endif
    }

}
