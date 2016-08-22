// Copyright (c) 2009-20512 The Regents of the University of Michigan
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
void export_union_sphere512(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 512> >(m, "IntegratorHPMCMonoSphereUnion512");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 512> >(m, "IntegratorHPMCMonoImplicitSphereUnion512");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 512> >(m, "ComputeFreeVolumeSphereUnion512");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 512> >(m, "AnalyzerSDFSphereUnion512");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 512> >(m, "UpdaterMuVTSphereUnion512");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 512> >(m, "UpdaterMuVTImplicitSphereUnion512");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 512> >(m, "ExternalFieldSphereUnion512");
    export_LatticeField<ShapeUnion<ShapeSphere, 512> >(m, "ExternalFieldLatticeSphereUnion512");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 512> >(m, "ExternalFieldCompositeSphereUnion512");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 512> >(m, "RemoveDriftUpdaterSphereUnion512");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 512> >(m, "WallSphereUnion512");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 512> >(m, "UpdaterExternalFieldWallSphereUnion512");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 512> >(m, "IntegratorHPMCMonoGPUSphereUnion512");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 512> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion512");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 512> >(m, "ComputeFreeVolumeGPUSphereUnion512");

    #endif
    }

}
