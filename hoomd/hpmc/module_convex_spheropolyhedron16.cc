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
void export_convex_spheropolyhedron16()
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoImplicitSpheropolyhedron16");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<16> >("ComputeFreeVolumeSpheropolyhedron16");
    export_AnalyzerSDF< ShapeSpheropolyhedron<16> >("AnalyzerSDFSpheropolyhedron16");
    export_UpdaterMuVT< ShapeSpheropolyhedron<16> >("UpdaterMuVTSpheropolyhedron16");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<16> >("UpdaterMuVTImplicitSpheropolyhedron16");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<16> >("ExternalFieldSpheropolyhedron16");
    export_LatticeField<ShapeSpheropolyhedron<16> >("ExternalFieldLatticeSpheropolyhedron16");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<16> >("ExternalFieldCompositeSpheropolyhedron16");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<16> >("RemoveDriftUpdaterSpheropolyhedron16");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<16> >("WallSpheropolyhedron16");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<16> >("UpdaterExternalFieldWallSpheropolyhedron16");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoGPUSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron16");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<16> >("ComputeFreeVolumeGPUSpheropolyhedron16");

    #endif
    #endif
    }

}
