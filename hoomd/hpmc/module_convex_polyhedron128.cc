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
void export_convex_polyhedron128()
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron<128> >("IntegratorHPMCMonoConvexPolyhedron128");
    export_IntegratorHPMCMonoImplicit< ShapeConvexPolyhedron<128> >("IntegratorHPMCMonoImplicitConvexPolyhedron128");
    export_ComputeFreeVolume< ShapeConvexPolyhedron<128> >("ComputeFreeVolumeConvexPolyhedron128");
    export_AnalyzerSDF< ShapeConvexPolyhedron<128> >("AnalyzerSDFConvexPolyhedron128");
    export_UpdaterMuVT< ShapeConvexPolyhedron<128> >("UpdaterMuVTConvexPolyhedron128");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<128> >("UpdaterMuVTImplicitConvexPolyhedron128");

    export_ExternalFieldInterface<ShapeConvexPolyhedron<128> >("ExternalFieldConvexPolyhedron128");
    export_LatticeField<ShapeConvexPolyhedron<128> >("ExternalFieldLatticeConvexPolyhedron128");
    export_ExternalFieldComposite<ShapeConvexPolyhedron<128> >("ExternalFieldCompositeConvexPolyhedron128");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron<128> >("RemoveDriftUpdaterConvexPolyhedron128");
    export_ExternalFieldWall<ShapeConvexPolyhedron<128> >("WallConvexPolyhedron128");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron<128> >("UpdaterExternalFieldWallConvexPolyhedron128");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron<128> >("IntegratorHPMCMonoGPUConvexPolyhedron128");
    export_IntegratorHPMCMonoImplicitGPU< ShapeConvexPolyhedron<128> >("IntegratorHPMCMonoImplicitGPUConvexPolyhedron128");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron<128> >("ComputeFreeVolumeGPUConvexPolyhedron128");

    #endif
    #endif
    }

}
