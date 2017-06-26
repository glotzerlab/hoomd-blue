// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeConvexPolyhedron.h"
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
void export_convex_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoConvexPolyhedron");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoImplicitConvexPolyhedron");
    export_ComputeFreeVolume< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeConvexPolyhedron");
    export_AnalyzerSDF< ShapeConvexPolyhedron >(m, "AnalyzerSDFConvexPolyhedron");
    export_UpdaterMuVT< ShapeConvexPolyhedron >(m, "UpdaterMuVTConvexPolyhedron");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron >(m, "UpdaterMuVTImplicitConvexPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron >(m, "ExternalFieldConvexPolyhedron");
    export_LatticeField<ShapeConvexPolyhedron >(m, "ExternalFieldLatticeConvexPolyhedron");
    export_ExternalFieldComposite<ShapeConvexPolyhedron >(m, "ExternalFieldCompositeConvexPolyhedron");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron >(m, "RemoveDriftUpdaterConvexPolyhedron");
    export_ExternalFieldWall<ShapeConvexPolyhedron >(m, "WallConvexPolyhedron");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron >(m, "UpdaterExternalFieldWallConvexPolyhedron");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoGPUConvexPolyhedron");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedron");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeGPUConvexPolyhedron");

    #endif
    }

}
