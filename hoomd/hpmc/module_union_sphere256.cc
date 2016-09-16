// Copyright (c) 2009-20256 The Regents of the University of Michigan
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
void export_union_sphere256(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 256> >(m, "IntegratorHPMCMonoSphereUnion256");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 256> >(m, "IntegratorHPMCMonoImplicitSphereUnion256");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 256> >(m, "ComputeFreeVolumeSphereUnion256");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 256> >(m, "AnalyzerSDFSphereUnion256");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 256> >(m, "UpdaterMuVTSphereUnion256");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 256> >(m, "UpdaterMuVTImplicitSphereUnion256");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 256> >(m, "ExternalFieldSphereUnion256");
    export_LatticeField<ShapeUnion<ShapeSphere, 256> >(m, "ExternalFieldLatticeSphereUnion256");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 256> >(m, "ExternalFieldCompositeSphereUnion256");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 256> >(m, "RemoveDriftUpdaterSphereUnion256");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 256> >(m, "WallSphereUnion256");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 256> >(m, "UpdaterExternalFieldWallSphereUnion256");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 256> >(m, "IntegratorHPMCMonoGPUSphereUnion256");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 256> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion256");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 256> >(m, "ComputeFreeVolumeGPUSphereUnion256");

    #endif
    }

}
