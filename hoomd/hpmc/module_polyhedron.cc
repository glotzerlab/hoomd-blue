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
void export_polyhedron()
    {
    export_IntegratorHPMCMono< ShapePolyhedron >("IntegratorHPMCMonoPolyhedron");
    export_IntegratorHPMCMonoImplicit< ShapePolyhedron >("IntegratorHPMCMonoImplicitPolyhedron");
    export_ComputeFreeVolume< ShapePolyhedron >("ComputeFreeVolumePolyhedron");
    // export_AnalyzerSDF< ShapePolyhedron >("AnalyzerSDFPolyhedron");
    export_UpdaterMuVT< ShapePolyhedron >("UpdaterMuVTPolyhedron");
    export_UpdaterMuVTImplicit< ShapePolyhedron >("UpdaterMuVTImplicitPolyhedron");

    export_ExternalFieldInterface<ShapePolyhedron>("ExternalFieldPolyhedron");
    export_LatticeField<ShapePolyhedron>("ExternalFieldLatticePolyhedron");
    export_ExternalFieldComposite<ShapePolyhedron>("ExternalFieldCompositePolyhedron");
    export_RemoveDriftUpdater<ShapePolyhedron>("RemoveDriftUpdaterPolyhedron");
    export_ExternalFieldWall<ShapePolyhedron>("WallPolyhedron");
    export_UpdaterExternalFieldWall<ShapePolyhedron>("UpdaterExternalFieldWallPolyhedron");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapePolyhedron >("IntegratorHPMCMonoGPUPolyhedron");
    export_IntegratorHPMCMonoImplicitGPU< ShapePolyhedron >("IntegratorHPMCMonoImplicitGPUPolyhedron");
    export_ComputeFreeVolumeGPU< ShapePolyhedron >("ComputeFreeVolumeGPUPolyhedron");
    #endif
    }

}
