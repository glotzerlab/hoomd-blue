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
void export_convex_polyhedron32()
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<32> >("IntegratorHPMCMonoConvexPolyhedron32");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<32> >("IntegratorHPMCMonoImplicitConvexPolyhedron32");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<32> >("ComputeFreeVolumeConvexPolyhedron32");
    export_AnalyzerSDF< ShapeConvexPolyhedron<32> >("AnalyzerSDFConvexPolyhedron32");
    export_UpdaterMuVT< ShapeConvexPolyhedron<32> >("UpdaterMuVTConvexPolyhedron32");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<32> >("UpdaterMuVTImplicitConvexPolyhedron32");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<32> >("ExternalFieldConvexPolyhedron32");
    export_LatticeField<ShapeConvexPolyhedron<32> >("ExternalFieldLatticeConvexPolyhedron32");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<32> >("ExternalFieldCompositeConvexPolyhedron32");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<32> >("RemoveDriftUpdaterConvexPolyhedron32");
    export_ExternalFieldWall<ShapeConvexPolyhedron<32> >("WallConvexPolyhedron32");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<32> >("UpdaterExternalFieldWallConvexPolyhedron32");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<32> >("IntegratorHPMCMonoGPUConvexPolyhedron32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<32> >("IntegratorHPMCMonoImplicitGPUConvexPolyhedron32");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<32> >("ComputeFreeVolumeGPUConvexPolyhedron32");

    #endif
    #endif
    }

}
