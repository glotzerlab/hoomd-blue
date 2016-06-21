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
void export_faceted_sphere()
    {
    export_IntegratorHPMCMono< ShapeFacetedSphere >("IntegratorHPMCMonoFacetedSphere");
    export_IntegratorHPMCMonoImplicit< ShapeFacetedSphere >("IntegratorHPMCMonoImplicitFacetedSphere");
    export_ComputeFreeVolume< ShapeFacetedSphere >("ComputeFreeVolumeFacetedSphere");
    export_AnalyzerSDF< ShapeFacetedSphere >("AnalyzerSDFFacetedSphere");
    export_UpdaterMuVT< ShapeFacetedSphere >("UpdaterMuVTFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere >("UpdaterMuVTImplicitFacetedSphere");

    export_ExternalFieldInterface<ShapeFacetedSphere>("ExternalFieldFacetedSphere");
    export_LatticeField<ShapeFacetedSphere>("ExternalFieldLatticeFacetedSphere");
    export_ExternalFieldComposite<ShapeFacetedSphere>("ExternalFieldCompositeFacetedSphere");
    export_RemoveDriftUpdater<ShapeFacetedSphere>("RemoveDriftUpdaterFacetedSphere");
    export_ExternalFieldWall<ShapeFacetedSphere>("WallFacetedSphere");
    export_UpdaterExternalFieldWall<ShapeFacetedSphere>("UpdaterExternalFieldWallFacetedSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeFacetedSphere >("IntegratorHPMCMonoGPUFacetedSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeFacetedSphere >("IntegratorHPMCMonoImplicitGPUFacetedSphere");
    export_ComputeFreeVolumeGPU< ShapeFacetedSphere >("ComputeFreeVolumeGPUFacetedSphere");
    #endif
    }

}
