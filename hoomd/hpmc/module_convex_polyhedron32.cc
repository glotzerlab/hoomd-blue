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
void export_convex_polyhedron32(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<32> >(m, "IntegratorHPMCMonoConvexPolyhedron32");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<32> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedron32");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<32> >(m, "ComputeFreeVolumeConvexPolyhedron32");
    export_AnalyzerSDF< ShapeConvexPolyhedron<32> >(m, "AnalyzerSDFConvexPolyhedron32");
    export_UpdaterMuVT< ShapeConvexPolyhedron<32> >(m, "UpdaterMuVTConvexPolyhedron32");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<32> >(m, "UpdaterMuVTImplicitConvexPolyhedron32");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<32> >(m, "ExternalFieldConvexPolyhedron32");
    export_LatticeField<ShapeConvexPolyhedron<32> >(m, "ExternalFieldLatticeConvexPolyhedron32");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<32> >(m, "ExternalFieldCompositeConvexPolyhedron32");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<32> >(m, "RemoveDriftUpdaterConvexPolyhedron32");
    export_ExternalFieldWall<ShapeConvexPolyhedron<32> >(m, "WallConvexPolyhedron32");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<32> >(m, "UpdaterExternalFieldWallConvexPolyhedron32");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<32> >(m, "IntegratorHPMCMonoGPUConvexPolyhedron32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<32> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedron32");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<32> >(m, "ComputeFreeVolumeGPUConvexPolyhedron32");

    #endif
    }

}
