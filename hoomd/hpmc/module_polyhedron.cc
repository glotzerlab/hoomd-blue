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
void export_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapePolyhedron >(m, "IntegratorHPMCMonoPolyhedron");
    export_IntegratorHPMCMonoImplicit< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitPolyhedron");
    export_ComputeFreeVolume< ShapePolyhedron >(m, "ComputeFreeVolumePolyhedron");
    // export_AnalyzerSDF< ShapePolyhedron >(m, "AnalyzerSDFPolyhedron");
    export_UpdaterMuVT< ShapePolyhedron >(m, "UpdaterMuVTPolyhedron");
    export_UpdaterMuVTImplicit< ShapePolyhedron >(m, "UpdaterMuVTImplicitPolyhedron");

    export_ExternalFieldInterface<ShapePolyhedron>(m, "ExternalFieldPolyhedron");
    export_LatticeField<ShapePolyhedron>(m, "ExternalFieldLatticePolyhedron");
    export_ExternalFieldComposite<ShapePolyhedron>(m, "ExternalFieldCompositePolyhedron");
    export_RemoveDriftUpdater<ShapePolyhedron>(m, "RemoveDriftUpdaterPolyhedron");
    export_ExternalFieldWall<ShapePolyhedron>(m, "WallPolyhedron");
    export_UpdaterExternalFieldWall<ShapePolyhedron>(m, "UpdaterExternalFieldWallPolyhedron");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapePolyhedron >(m, "IntegratorHPMCMonoGPUPolyhedron");
    export_IntegratorHPMCMonoImplicitGPU< ShapePolyhedron >(m, "IntegratorHPMCMonoImplicitGPUPolyhedron");
    export_ComputeFreeVolumeGPU< ShapePolyhedron >(m, "ComputeFreeVolumeGPUPolyhedron");
    #endif
    }

}
