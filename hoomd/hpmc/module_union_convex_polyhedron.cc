// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"
#include "AnalyzerSDF.h"

#include "ShapeUnion.h"
#include "ShapeSpheropolyhedron.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

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
void export_union_convex_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSpheropolyhedron> >(m, "IntegratorHPMCMonoConvexPolyhedronUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeSpheropolyhedron> >(m, "ComputeFreeVolumeConvexPolyhedronUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeSpheropolyhedron> >(m, "AnalyzerSDFConvexPolyhedronUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSpheropolyhedron> >(m, "UpdaterMuVTConvexPolyhedronUnion");
    export_UpdaterClusters<ShapeUnion<ShapeSpheropolyhedron> >(m, "UpdaterClustersConvexPolyhedronUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSpheropolyhedron> >(m, "ExternalFieldConvexPolyhedronUnion");
    export_LatticeField<ShapeUnion<ShapeSpheropolyhedron> >(m, "ExternalFieldLatticeConvexPolyhedronUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeSpheropolyhedron> >(m, "ExternalFieldCompositeConvexPolyhedronUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSpheropolyhedron> >(m, "RemoveDriftUpdaterConvexPolyhedronUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSpheropolyhedron> >(m, "WallConvexPolyhedronUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSpheropolyhedron> >(m, "UpdaterExternalFieldWallConvexPolyhedronUnion");

    #ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSpheropolyhedron> >(m, "IntegratorHPMCMonoGPUConvexPolyhedronUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSpheropolyhedron> >(m, "ComputeFreeVolumeGPUConvexPolyhedronUnion");

    #endif
    }

}
