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
void export_union_sphere()
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere> >("IntegratorHPMCMonoSphere");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere> >("IntegratorHPMCMonoImplicitSphere");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere> >("ComputeFreeVolumeSphere");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere> >("AnalyzerSDFSphere");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere> >("UpdaterMuVTSphere");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere> >("UpdaterMuVTImplicitSphere");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere> >("ExternalFieldSphere");
    export_LatticeField<ShapeUnion<ShapeSphere> >("ExternalFieldLatticeSphere");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere> >("ExternalFieldCompositeSphere");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere> >("RemoveDriftUpdaterSphere");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere> >("WallSphere");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere> >("UpdaterExternalFieldWallSphere");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere> >("IntegratorHPMCMonoGPUSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere> >("IntegratorHPMCMonoImplicitGPUSphere");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere> >("ComputeFreeVolumeGPUSphere");

    #endif
    #endif
    }

}
