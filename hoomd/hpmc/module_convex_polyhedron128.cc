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
void export_convex_polyhedron128(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<128> >(m, "IntegratorHPMCMonoConvexPolyhedron128");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<128> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedron128");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<128> >(m, "ComputeFreeVolumeConvexPolyhedron128");
    export_AnalyzerSDF< ShapeConvexPolyhedron<128> >(m, "AnalyzerSDFConvexPolyhedron128");
    export_UpdaterMuVT< ShapeConvexPolyhedron<128> >(m, "UpdaterMuVTConvexPolyhedron128");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<128> >(m, "UpdaterMuVTImplicitConvexPolyhedron128");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<128> >(m, "ExternalFieldConvexPolyhedron128");
    export_LatticeField<ShapeConvexPolyhedron<128> >(m, "ExternalFieldLatticeConvexPolyhedron128");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<128> >(m, "ExternalFieldCompositeConvexPolyhedron128");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<128> >(m, "RemoveDriftUpdaterConvexPolyhedron128");
    export_ExternalFieldWall<ShapeConvexPolyhedron<128> >(m, "WallConvexPolyhedron128");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<128> >(m, "UpdaterExternalFieldWallConvexPolyhedron128");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<128> >(m, "IntegratorHPMCMonoGPUConvexPolyhedron128");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<128> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedron128");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<128> >(m, "ComputeFreeVolumeGPUConvexPolyhedron128");

    #endif
    }

}
