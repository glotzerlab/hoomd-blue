// Copyright (c) 2009-20128 The Regents of the University of Michigan
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
void export_union_sphere128(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 128> >(m, "IntegratorHPMCMonoSphereUnion128");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 128> >(m, "IntegratorHPMCMonoImplicitSphereUnion128");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 128> >(m, "ComputeFreeVolumeSphereUnion128");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 128> >(m, "AnalyzerSDFSphereUnion128");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 128> >(m, "UpdaterMuVTSphereUnion128");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 128> >(m, "UpdaterMuVTImplicitSphereUnion128");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 128> >(m, "ExternalFieldSphereUnion128");
    export_LatticeField<ShapeUnion<ShapeSphere, 128> >(m, "ExternalFieldLatticeSphereUnion128");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 128> >(m, "ExternalFieldCompositeSphereUnion128");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 128> >(m, "RemoveDriftUpdaterSphereUnion128");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 128> >(m, "WallSphereUnion128");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 128> >(m, "UpdaterExternalFieldWallSphereUnion128");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 128> >(m, "IntegratorHPMCMonoGPUSphereUnion128");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 128> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion128");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 128> >(m, "ComputeFreeVolumeGPUSphereUnion128");

    #endif
    }

}
