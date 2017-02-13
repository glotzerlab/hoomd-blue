// Copyright (c) 2009-2017 The Regents of the University of Michigan
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
void export_union_sphere16(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 16> >(m, "IntegratorHPMCMonoSphereUnion16");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 16> >(m, "IntegratorHPMCMonoImplicitSphereUnion16");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 16> >(m, "ComputeFreeVolumeSphereUnion16");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 16, > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 16> >(m, "UpdaterMuVTSphereUnion16");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 16> >(m, "UpdaterMuVTImplicitSphereUnion16");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 16> >(m, "ExternalFieldSphereUnion16");
    export_LatticeField<ShapeUnion<ShapeSphere, 16> >(m, "ExternalFieldLatticeSphereUnion16");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 16> >(m, "ExternalFieldCompositeSphereUnion16");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 16> >(m, "RemoveDriftUpdaterSphereUnion16");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 16> >(m, "WallSphereUnion16");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 16> >(m, "UpdaterExternalFieldWallSphereUnion16");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 16> >(m, "IntegratorHPMCMonoGPUSphereUnion16");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 16> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion16");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 16> >(m, "ComputeFreeVolumeGPUSphereUnion16");

    #endif
    }

}
