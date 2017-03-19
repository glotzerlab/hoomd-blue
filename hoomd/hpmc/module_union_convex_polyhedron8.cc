// Copyright (c) 2009-2017 The Regents of the University of Michigan
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
void export_union_convex_polyhedron8(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "IntegratorHPMCMonoConvexPolyhedronUnion8");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "IntegratorHPMCMonoImplicitConvexPolyhedronUnion8");
    export_ComputeFreeVolume< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "ComputeFreeVolumeConvexPolyhedronUnion8");
    // export_AnalyzerSDF< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "AnalyzerSDFConvexPolyhedronUnion8");
    export_UpdaterMuVT< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "UpdaterMuVTConvexPolyhedronUnion8");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "UpdaterMuVTImplicitConvexPolyhedronUnion8");

    export_ExternalFieldInterface<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "ExternalFieldConvexPolyhedronUnion8");
    export_LatticeField<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "ExternalFieldLatticeConvexPolyhedronUnion8");
    export_ExternalFieldComposite<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "ExternalFieldCompositeConvexPolyhedronUnion8");
    export_RemoveDriftUpdater<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "RemoveDriftUpdaterConvexPolyhedronUnion8");
    export_ExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "WallConvexPolyhedronUnion8");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "UpdaterExternalFieldWallConvexPolyhedronUnion8");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "IntegratorHPMCMonoGPUConvexPolyhedronUnion8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "IntegratorHPMCMonoImplicitGPUConvexPolyhedronUnion8");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeConvexPolyhedron<128>, 8> >(m, "ComputeFreeVolumeGPUConvexPolyhedronUnion8");

    #endif
    }

}
