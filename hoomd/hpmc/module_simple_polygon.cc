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
void export_simple_polygon()
    {
    export_IntegratorHPMCMono< ShapeSimplePolygon >("IntegratorHPMCMonoSimplePolygon");
    export_IntegratorHPMCMonoImplicit< ShapeSimplePolygon >("IntegratorHPMCMonoImplicitSimplePolygon");
    export_ComputeFreeVolume< ShapeSimplePolygon >("ComputeFreeVolumeSimplePolygon");
    export_AnalyzerSDF< ShapeSimplePolygon >("AnalyzerSDFSimplePolygon");
    export_UpdaterMuVT< ShapeSimplePolygon >("UpdaterMuVTSimplePolygon");
    export_UpdaterMuVTImplicit< ShapeSimplePolygon >("UpdaterMuVTImplicitSimplePolygon");

    export_ExternalFieldInterface<ShapeSimplePolygon>("ExternalFieldSimplePolygon");
    export_LatticeField<ShapeSimplePolygon>("ExternalFieldLatticeSimplePolygon");
    export_ExternalFieldComposite<ShapeSimplePolygon>("ExternalFieldCompositeSimplePolygon");
    export_RemoveDriftUpdater<ShapeSimplePolygon>("RemoveDriftUpdaterSimplePolygon");
    // export_ExternalFieldWall<ShapeSimplePolygon>("WallSimplePolygon");
    // export_UpdaterExternalFieldWall<ShapeSimplePolygon>("UpdaterExternalFieldWallSimplePolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeSimplePolygon >("IntegratorHPMCMonoGPUSimplePolygon");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSimplePolygon >("IntegratorHPMCMonoImplicitGPUSimplePolygon");
    export_ComputeFreeVolumeGPU< ShapeSimplePolygon >("ComputeFreeVolumeGPUSimplePolygon");
    #endif
    }

}
