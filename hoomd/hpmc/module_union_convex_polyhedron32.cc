// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
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

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif

namespace py = pybind11;

using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_union_convex_polyhedron32(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "IntegratorHPMCMonoConvexPolyhedronUnion32");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedronUnion32");
    export_ComputeFreeVolume< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "ComputeFreeVolumeConvexPolyhedronUnion32");
    // export_AnalyzerSDF< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "AnalyzerSDFConvexPolyhedronUnion32");
    export_UpdaterMuVT< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "UpdaterMuVTConvexPolyhedronUnion32");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "UpdaterMuVTImplicitConvexPolyhedronUnion32");

    export_ExternalFieldInterface<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "ExternalFieldConvexPolyhedronUnion32");
    export_LatticeField<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "ExternalFieldLatticeConvexPolyhedronUnion32");
    export_ExternalFieldComposite<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "ExternalFieldCompositeConvexPolyhedronUnion32");
    export_RemoveDriftUpdater<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "RemoveDriftUpdaterConvexPolyhedronUnion32");
    export_ExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "WallConvexPolyhedronUnion32");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "UpdaterExternalFieldWallConvexPolyhedronUnion32");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "IntegratorHPMCMonoGPUConvexPolyhedronUnion32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedronUnion32");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeConvexPolyhedron, 32> >(m, "ComputeFreeVolumeGPUConvexPolyhedronUnion32");

    #endif
    }

}
