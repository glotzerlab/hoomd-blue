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
void export_convex_polyhedron64(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<64> >(m, "IntegratorHPMCMonoConvexPolyhedron64");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<64> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedron64");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<64> >(m, "ComputeFreeVolumeConvexPolyhedron64");
    export_AnalyzerSDF< ShapeConvexPolyhedron<64> >(m, "AnalyzerSDFConvexPolyhedron64");
    export_UpdaterMuVT< ShapeConvexPolyhedron<64> >(m, "UpdaterMuVTConvexPolyhedron64");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<64> >(m, "UpdaterMuVTImplicitConvexPolyhedron64");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<64> >(m, "ExternalFieldConvexPolyhedron64");
    export_LatticeField<ShapeConvexPolyhedron<64> >(m, "ExternalFieldLatticeConvexPolyhedron64");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<64> >(m, "ExternalFieldCompositeConvexPolyhedron64");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<64> >(m, "RemoveDriftUpdaterConvexPolyhedron64");
    export_ExternalFieldWall<ShapeConvexPolyhedron<64> >(m, "WallConvexPolyhedron64");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<64> >(m, "UpdaterExternalFieldWallConvexPolyhedron64");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<64> >(m, "IntegratorHPMCMonoGPUConvexPolyhedron64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<64> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedron64");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<64> >(m, "ComputeFreeVolumeGPUConvexPolyhedron64");

    #endif
    }

}
