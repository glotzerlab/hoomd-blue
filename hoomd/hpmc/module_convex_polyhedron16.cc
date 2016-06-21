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
void export_convex_polyhedron16()
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<16> >("IntegratorHPMCMonoConvexPolyhedron16");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<16> >("IntegratorHPMCMonoImplicitConvexPolyhedron16");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<16> >("ComputeFreeVolumeConvexPolyhedron16");
    export_AnalyzerSDF< ShapeConvexPolyhedron<16> >("AnalyzerSDFConvexPolyhedron16");
    export_UpdaterMuVT< ShapeConvexPolyhedron<16> >("UpdaterMuVTConvexPolyhedron16");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<16> >("UpdaterMuVTImplicitConvexPolyhedron16");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<16> >("ExternalFieldConvexPolyhedron16");
    export_LatticeField<ShapeConvexPolyhedron<16> >("ExternalFieldLatticeConvexPolyhedron16");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<16> >("ExternalFieldCompositeConvexPolyhedron16");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<16> >("RemoveDriftUpdaterConvexPolyhedron16");
    export_ExternalFieldWall<ShapeConvexPolyhedron<16> >("WallConvexPolyhedron16");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<16> >("UpdaterExternalFieldWallConvexPolyhedron16");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<16> >("IntegratorHPMCMonoGPUConvexPolyhedron16");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<16> >("IntegratorHPMCMonoImplicitGPUConvexPolyhedron16");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<16> >("ComputeFreeVolumeGPUConvexPolyhedron16");

    #endif
    }

}
