// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"
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
void export_union_sphere4(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 4> >(m, "IntegratorHPMCMonoSphereUnion4");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 4> >(m, "IntegratorHPMCMonoImplicitSphereUnion4");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 4> >(m, "ComputeFreeVolumeSphereUnion4");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 4, > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 4> >(m, "UpdaterMuVTSphereUnion4");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 4> >(m, "UpdaterMuVTImplicitSphereUnion4");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 4> >(m, "ExternalFieldSphereUnion4");
    export_LatticeField<ShapeUnion<ShapeSphere, 4> >(m, "ExternalFieldLatticeSphereUnion4");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 4> >(m, "ExternalFieldCompositeSphereUnion4");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 4> >(m, "RemoveDriftUpdaterSphereUnion4");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 4> >(m, "WallSphereUnion4");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 4> >(m, "UpdaterExternalFieldWallSphereUnion4");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 4> >(m, "IntegratorHPMCMonoGPUSphereUnion4");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 4> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion4");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 4> >(m, "ComputeFreeVolumeGPUSphereUnion4");

    #endif
    }

}
