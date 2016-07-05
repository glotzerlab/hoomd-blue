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




namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_simple_polygon(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSimplePolygon >(m, "IntegratorHPMCMonoSimplePolygon");
    export_IntegratorHPMCMonoImplicit< ShapeSimplePolygon >(m, "IntegratorHPMCMonoImplicitSimplePolygon");
    export_ComputeFreeVolume< ShapeSimplePolygon >(m, "ComputeFreeVolumeSimplePolygon");
    export_AnalyzerSDF< ShapeSimplePolygon >(m, "AnalyzerSDFSimplePolygon");
    export_UpdaterMuVT< ShapeSimplePolygon >(m, "UpdaterMuVTSimplePolygon");
    export_UpdaterMuVTImplicit< ShapeSimplePolygon >(m, "UpdaterMuVTImplicitSimplePolygon");

    export_ExternalFieldInterface<ShapeSimplePolygon>(m, "ExternalFieldSimplePolygon");
    export_LatticeField<ShapeSimplePolygon>(m, "ExternalFieldLatticeSimplePolygon");
    export_ExternalFieldComposite<ShapeSimplePolygon>(m, "ExternalFieldCompositeSimplePolygon");
    export_RemoveDriftUpdater<ShapeSimplePolygon>(m, "RemoveDriftUpdaterSimplePolygon");
    // export_ExternalFieldWall<ShapeSimplePolygon>(m, "WallSimplePolygon");
    // export_UpdaterExternalFieldWall<ShapeSimplePolygon>(m, "UpdaterExternalFieldWallSimplePolygon");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeSimplePolygon >(m, "IntegratorHPMCMonoGPUSimplePolygon");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSimplePolygon >(m, "IntegratorHPMCMonoImplicitGPUSimplePolygon");
    export_ComputeFreeVolumeGPU< ShapeSimplePolygon >(m, "ComputeFreeVolumeGPUSimplePolygon");
    #endif
    }

}
