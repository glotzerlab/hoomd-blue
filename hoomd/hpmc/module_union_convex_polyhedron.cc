// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "IntegratorHPMCMonoImplicitNew.h"
#include "ComputeFreeVolume.h"
#include "AnalyzerSDF.h"

#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

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
void export_union_convex_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoConvexPolyhedronUnion");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedronUnion");
    export_IntegratorHPMCMonoImplicitNew< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoImplicitNewConvexPolyhedronUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeConvexPolyhedron> >(m, "ComputeFreeVolumeConvexPolyhedronUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeConvexPolyhedron> >(m, "AnalyzerSDFConvexPolyhedronUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterMuVTConvexPolyhedronUnion");
    export_UpdaterClusters<ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterClustersConvexPolyhedronUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeConvexPolyhedron>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterClustersImplicitConvexPolyhedronUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeConvexPolyhedron>, IntegratorHPMCMonoImplicitNew<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterClustersImplicitNewConvexPolyhedronUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterMuVTImplicitConvexPolyhedronUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron>, IntegratorHPMCMonoImplicitNew<ShapeUnion<ShapeConvexPolyhedron> > >(m, "UpdaterMuVTImplicitNewConvexPolyhedronUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldConvexPolyhedronUnion");
    export_LatticeField<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldLatticeConvexPolyhedronUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeConvexPolyhedron> >(m, "ExternalFieldCompositeConvexPolyhedronUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeConvexPolyhedron> >(m, "RemoveDriftUpdaterConvexPolyhedronUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron> >(m, "WallConvexPolyhedronUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron> >(m, "UpdaterExternalFieldWallConvexPolyhedronUnion");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoGPUConvexPolyhedronUnion");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedronUnion");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "IntegratorHPMCMonoImplicitNewGPUConvexPolyhedronUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeConvexPolyhedron> >(m, "ComputeFreeVolumeGPUConvexPolyhedronUnion");

    #endif
    }

}
