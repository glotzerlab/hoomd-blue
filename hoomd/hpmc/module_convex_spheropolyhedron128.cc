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
void export_convex_spheropolyhedron128()
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoSpheropolyhedron128");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoImplicitSpheropolyhedron128");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<128> >("ComputeFreeVolumeSpheropolyhedron128");
    export_AnalyzerSDF< ShapeSpheropolyhedron<128> >("AnalyzerSDFSpheropolyhedron128");
    export_UpdaterMuVT< ShapeSpheropolyhedron<128> >("UpdaterMuVTSpheropolyhedron128");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<128> >("UpdaterMuVTImplicitSpheropolyhedron128");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<128> >("ExternalFieldSpheropolyhedron128");
    export_LatticeField<ShapeSpheropolyhedron<128> >("ExternalFieldLatticeSpheropolyhedron128");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<128> >("ExternalFieldCompositeSpheropolyhedron128");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<128> >("RemoveDriftUpdaterSpheropolyhedron128");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<128> >("WallSpheropolyhedron128");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<128> >("UpdaterExternalFieldWallSpheropolyhedron128");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoGPUSpheropolyhedron128");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron128");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<128> >("ComputeFreeVolumeGPUSpheropolyhedron128");

    #endif
    }

}
