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
void export_convex_polyhedron8(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<8> >(m, "IntegratorHPMCMonoConvexPolyhedron8");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<8> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedron8");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<8> >(m, "ComputeFreeVolumeConvexPolyhedron8");
    export_AnalyzerSDF< ShapeConvexPolyhedron<8> >(m, "AnalyzerSDFConvexPolyhedron8");
    export_UpdaterMuVT< ShapeConvexPolyhedron<8> >(m, "UpdaterMuVTConvexPolyhedron8");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<8> >(m, "UpdaterMuVTImplicitConvexPolyhedron8");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<8> >(m, "ExternalFieldConvexPolyhedron8");
    export_LatticeField<ShapeConvexPolyhedron<8> >(m, "ExternalFieldLatticeConvexPolyhedron8");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<8> >(m, "ExternalFieldCompositeConvexPolyhedron8");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<8> >(m, "RemoveDriftUpdaterConvexPolyhedron8");
    export_ExternalFieldWall<ShapeConvexPolyhedron<8> >(m, "WallConvexPolyhedron8");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<8> >(m, "UpdaterExternalFieldWallConvexPolyhedron8");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<8> >(m, "IntegratorHPMCMonoGPUConvexPolyhedron8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<8> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedron8");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<8> >(m, "ComputeFreeVolumeGPUConvexPolyhedron8");

    #endif
    }

}
