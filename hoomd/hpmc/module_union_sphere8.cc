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
void export_union_sphere8(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphere, 8> >(m, "IntegratorHPMCMonoSphereUnion8");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphere, 8> >(m, "IntegratorHPMCMonoImplicitSphereUnion8");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphere, 8> >(m, "ComputeFreeVolumeSphereUnion8");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphere, 8, > >(m, "AnalyzerSDFSphereUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere, 8> >(m, "UpdaterMuVTSphereUnion8");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere, 8> >(m, "UpdaterMuVTImplicitSphereUnion8");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere, 8> >(m, "ExternalFieldSphereUnion8");
    export_LatticeField<ShapeUnion<ShapeSphere, 8> >(m, "ExternalFieldLatticeSphereUnion8");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere, 8> >(m, "ExternalFieldCompositeSphereUnion8");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphere, 8> >(m, "RemoveDriftUpdaterSphereUnion8");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere, 8> >(m, "WallSphereUnion8");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere, 8> >(m, "UpdaterExternalFieldWallSphereUnion8");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphere, 8> >(m, "IntegratorHPMCMonoGPUSphereUnion8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphere, 8> >(m, "IntegratorHPMCMonoImplicitGPUSphereUnion8");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphere, 8> >(m, "ComputeFreeVolumeGPUSphereUnion8");

    #endif
    }

}
