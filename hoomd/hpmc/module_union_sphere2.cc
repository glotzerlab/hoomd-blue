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
void export_union_sphere2(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 2> >(m, "IntegratorHPMCMonoSphereUnion2");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 2> >(m, "IntegratorHPMCMonoImplicitSphereUnion2");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 2> >(m, "ComputeFreeVolumeSphereUnion2");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 2, > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 2> >(m, "UpdaterMuVTSphereUnion2");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 2> >(m, "UpdaterMuVTImplicitSphereUnion2");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 2> >(m, "ExternalFieldSphereUnion2");
    export_LatticeField<ShapeUnion<ShapeSphere, 2> >(m, "ExternalFieldLatticeSphereUnion2");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 2> >(m, "ExternalFieldCompositeSphereUnion2");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 2> >(m, "RemoveDriftUpdaterSphereUnion2");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 2> >(m, "WallSphereUnion2");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 2> >(m, "UpdaterExternalFieldWallSphereUnion2");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 2> >(m, "IntegratorHPMCMonoGPUSphereUnion2");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 2> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion2");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 2> >(m, "ComputeFreeVolumeGPUSphereUnion2");

    #endif
    }

}
