// Copyright (c) 2009-2032 The Regents of the University of Michigan
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
void export_union_sphere32(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 32> >(m, "IntegratorHPMCMonoSphereUnion32");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 32> >(m, "IntegratorHPMCMonoImplicitSphereUnion32");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 32> >(m, "ComputeFreeVolumeSphereUnion32");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 32> >(m, "AnalyzerSDFSphereUnion32");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 32> >(m, "UpdaterMuVTSphereUnion32");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 32> >(m, "UpdaterMuVTImplicitSphereUnion32");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 32> >(m, "ExternalFieldSphereUnion32");
    export_LatticeField<ShapeUnion<ShapeSphere, 32> >(m, "ExternalFieldLatticeSphereUnion32");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 32> >(m, "ExternalFieldCompositeSphereUnion32");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 32> >(m, "RemoveDriftUpdaterSphereUnion32");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 32> >(m, "WallSphereUnion32");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 32> >(m, "UpdaterExternalFieldWallSphereUnion32");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 32> >(m, "IntegratorHPMCMonoGPUSphereUnion32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 32> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion32");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 32> >(m, "ComputeFreeVolumeGPUSphereUnion32");

    #endif
    }

}
