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
void export_convex_polyhedron8()
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<8> >("IntegratorHPMCMonoConvexPolyhedron8");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<8> >("IntegratorHPMCMonoImplicitConvexPolyhedron8");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<8> >("ComputeFreeVolumeConvexPolyhedron8");
    export_AnalyzerSDF< ShapeConvexPolyhedron<8> >("AnalyzerSDFConvexPolyhedron8");
    export_UpdaterMuVT< ShapeConvexPolyhedron<8> >("UpdaterMuVTConvexPolyhedron8");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<8> >("UpdaterMuVTImplicitConvexPolyhedron8");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<8> >("ExternalFieldConvexPolyhedron8");
    export_LatticeField<ShapeConvexPolyhedron<8> >("ExternalFieldLatticeConvexPolyhedron8");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<8> >("ExternalFieldCompositeConvexPolyhedron8");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<8> >("RemoveDriftUpdaterConvexPolyhedron8");
    export_ExternalFieldWall<ShapeConvexPolyhedron<8> >("WallConvexPolyhedron8");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<8> >("UpdaterExternalFieldWallConvexPolyhedron8");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<8> >("IntegratorHPMCMonoGPUConvexPolyhedron8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<8> >("IntegratorHPMCMonoImplicitGPUConvexPolyhedron8");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<8> >("ComputeFreeVolumeGPUConvexPolyhedron8");

    #endif
    }

}
