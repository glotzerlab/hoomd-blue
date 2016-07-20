// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
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
void export_convex_spheropolyhedron16(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<16> >(m, "IntegratorHPMCMonoSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<16> >(m, "IntegratorHPMCMonoImplicitSpheropolyhedron16");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<16> >(m, "ComputeFreeVolumeSpheropolyhedron16");
    export_AnalyzerSDF< ShapeSpheropolyhedron<16> >(m, "AnalyzerSDFSpheropolyhedron16");
    export_UpdaterMuVT< ShapeSpheropolyhedron<16> >(m, "UpdaterMuVTSpheropolyhedron16");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<16> >(m, "UpdaterMuVTImplicitSpheropolyhedron16");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<16> >(m, "ExternalFieldSpheropolyhedron16");
    export_LatticeField<ShapeSpheropolyhedron<16> >(m, "ExternalFieldLatticeSpheropolyhedron16");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<16> >(m, "ExternalFieldCompositeSpheropolyhedron16");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<16> >(m, "RemoveDriftUpdaterSpheropolyhedron16");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<16> >(m, "WallSpheropolyhedron16");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<16> >(m, "UpdaterExternalFieldWallSpheropolyhedron16");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<16> >(m, "IntegratorHPMCMonoGPUSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<16> >(m, "IntegratorHPMCMonoImplicitGPUSpheropolyhedron16");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<16> >(m, "ComputeFreeVolumeGPUSpheropolyhedron16");

    #endif
    }

}
