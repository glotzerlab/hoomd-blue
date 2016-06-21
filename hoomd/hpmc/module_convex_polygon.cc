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
    export_IntegratorHPMCMono< ShapeConvexPolygon >("IntegratorHPMCMonoSphere");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolygon >("IntegratorHPMCMonoImplicitSphere");
    export_ComputeFreeVolume< ShapeConvexPolygon >("ComputeFreeVolumeSphere");
    export_AnalyzerSDF< ShapeConvexPolygon >("AnalyzerSDFSphere");
    export_UpdaterMuVT< ShapeConvexPolygon >("UpdaterMuVTSphere");
    export_UpdaterMuVTImplicit< ShapeConvexPolygon >("UpdaterMuVTImplicitSphere");

    export_ExternalFieldInterface<ShapeConvexPolygon>("ExternalFieldSphere");
    export_LatticeField<ShapeConvexPolygon>("ExternalFieldLatticeSphere");
    export_ExternalFieldComposite<ShapeConvexPolygon>("ExternalFieldCompositeSphere");
    export_RemoveDriftUpdater<ShapeConvexPolygon>("RemoveDriftUpdaterSphere");
    // export_ExternalFieldWall<ShapeConvexPolygon>("WallSphere");
    // export_UpdaterExternalFieldWall<ShapeConvexPolygon>("UpdaterExternalFieldWallSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeConvexPolygon >("IntegratorHPMCMonoGPUSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolygon >("IntegratorHPMCMonoImplicitGPUSphere");
    export_ComputeFreeVolumeGPU< ShapeConvexPolygon >("ComputeFreeVolumeGPUSphere");
    #endif
    }

}
