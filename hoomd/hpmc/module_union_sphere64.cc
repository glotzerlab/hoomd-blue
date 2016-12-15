// Copyright (c) 2009-2064 The Regents of the University of Michigan
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
void export_union_sphere64(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 64> >(m, "IntegratorHPMCMonoSphereUnion64");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 64> >(m, "IntegratorHPMCMonoImplicitSphereUnion64");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 64> >(m, "ComputeFreeVolumeSphereUnion64");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 64> >(m, "AnalyzerSDFSphereUnion64");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 64> >(m, "UpdaterMuVTSphereUnion64");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 64> >(m, "UpdaterMuVTImplicitSphereUnion64");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 64> >(m, "ExternalFieldSphereUnion64");
    export_LatticeField<ShapeUnion<ShapeSphere, 64> >(m, "ExternalFieldLatticeSphereUnion64");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 64> >(m, "ExternalFieldCompositeSphereUnion64");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 64> >(m, "RemoveDriftUpdaterSphereUnion64");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 64> >(m, "WallSphereUnion64");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 64> >(m, "UpdaterExternalFieldWallSphereUnion64");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 64> >(m, "IntegratorHPMCMonoGPUSphereUnion64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 64> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion64");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 64> >(m, "ComputeFreeVolumeGPUSphereUnion64");

    #endif
    }

}
