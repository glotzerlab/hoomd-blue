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
void export_convex_spheropolyhedron64()
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoImplicitSpheropolyhedron64");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<64> >("ComputeFreeVolumeSpheropolyhedron64");
    export_AnalyzerSDF< ShapeSpheropolyhedron<64> >("AnalyzerSDFSpheropolyhedron64");
    export_UpdaterMuVT< ShapeSpheropolyhedron<64> >("UpdaterMuVTSpheropolyhedron64");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<64> >("UpdaterMuVTImplicitSpheropolyhedron64");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<64> >("ExternalFieldSpheropolyhedron64");
    export_LatticeField<ShapeSpheropolyhedron<64> >("ExternalFieldLatticeSpheropolyhedron64");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<64> >("ExternalFieldCompositeSpheropolyhedron64");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<64> >("RemoveDriftUpdaterSpheropolyhedron64");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<64> >("WallSpheropolyhedron64");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<64> >("UpdaterExternalFieldWallSpheropolyhedron64");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoGPUSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron64");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<64> >("ComputeFreeVolumeGPUSpheropolyhedron64");

    #endif
    }

}
