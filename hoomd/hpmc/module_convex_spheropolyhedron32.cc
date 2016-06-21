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
void export_convex_spheropolyhedron32()
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoImplicitSpheropolyhedron32");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<32> >("ComputeFreeVolumeSpheropolyhedron32");
    export_AnalyzerSDF< ShapeSpheropolyhedron<32> >("AnalyzerSDFSpheropolyhedron32");
    export_UpdaterMuVT< ShapeSpheropolyhedron<32> >("UpdaterMuVTSpheropolyhedron32");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<32> >("UpdaterMuVTImplicitSpheropolyhedron32");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<32> >("ExternalFieldSpheropolyhedron32");
    export_LatticeField<ShapeSpheropolyhedron<32> >("ExternalFieldLatticeSpheropolyhedron32");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<32> >("ExternalFieldCompositeSpheropolyhedron32");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<32> >("RemoveDriftUpdaterSpheropolyhedron32");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<32> >("WallSpheropolyhedron32");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<32> >("UpdaterExternalFieldWallSpheropolyhedron32");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoGPUSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron32");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<32> >("ComputeFreeVolumeGPUSpheropolyhedron32");

    #endif
    }

}
