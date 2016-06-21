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

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_convex_polygon()
    {
    export_IntegratorHPMCMono< ShapeConvexPolygon >("IntegratorHPMCMonoConvexPolygon");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolygon >("IntegratorHPMCMonoImplicitConvexPolygon");
    export_ComputeFreeVolume< ShapeConvexPolygon >("ComputeFreeVolumeConvexPolygon");
    export_AnalyzerSDF< ShapeConvexPolygon >("AnalyzerSDFConvexPolygon");
    export_UpdaterMuVT< ShapeConvexPolygon >("UpdaterMuVTConvexPolygon");
    export_UpdaterMuVTImplicit< ShapeConvexPolygon >("UpdaterMuVTImplicitConvexPolygon");

    export_ExternalFieldInterface<ShapeConvexPolygon>("ExternalFieldConvexPolygon");
    export_LatticeField<ShapeConvexPolygon>("ExternalFieldLatticeConvexPolygon");
    export_ExternalFieldComposite<ShapeConvexPolygon>("ExternalFieldCompositeConvexPolygon");
    export_RemoveDriftUpdater<ShapeConvexPolygon>("RemoveDriftUpdaterConvexPolygon");
    // export_ExternalFieldWall<ShapeConvexPolygon>("WallConvexPolygon");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>("UpdaterExternalFieldWallConvexPolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeConvexPolygon >("IntegratorHPMCMonoGPUConvexPolygon");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolygon >("IntegratorHPMCMonoImplicitGPUConvexPolygon");
    export_ComputeFreeVolumeGPU< ShapeConvexPolygon >("ComputeFreeVolumeGPUConvexPolygon");
    #endif
    }

}
