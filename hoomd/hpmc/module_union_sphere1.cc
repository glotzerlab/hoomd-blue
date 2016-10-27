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
void export_union_sphere1(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 1> >(m, "IntegratorHPMCMonoSphereUnion1");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 1> >(m, "IntegratorHPMCMonoImplicitSphereUnion1");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 1> >(m, "ComputeFreeVolumeSphereUnion1");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 1, > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 1> >(m, "UpdaterMuVTSphereUnion1");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 1> >(m, "UpdaterMuVTImplicitSphereUnion1");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 1> >(m, "ExternalFieldSphereUnion1");
    export_LatticeField<ShapeUnion<ShapeSphere, 1> >(m, "ExternalFieldLatticeSphereUnion1");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 1> >(m, "ExternalFieldCompositeSphereUnion1");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 1> >(m, "RemoveDriftUpdaterSphereUnion1");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 1> >(m, "WallSphereUnion1");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 1> >(m, "UpdaterExternalFieldWallSphereUnion1");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 1> >(m, "IntegratorHPMCMonoGPUSphereUnion1");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 1> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion1");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 1> >(m, "ComputeFreeVolumeGPUSphereUnion1");

    #endif
    }

}
