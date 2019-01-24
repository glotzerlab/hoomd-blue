// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolyhedron.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"
#include "ExternalCallback.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "IntegratorHPMCMonoImplicitNewGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif




namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_convex_spheropolyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoSpheropolyhedron");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoImplicitSpheropolyhedron");
    export_ComputeFreeVolume< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeSpheropolyhedron");
    export_AnalyzerSDF< ShapeSpheropolyhedron >(m, "AnalyzerSDFSpheropolyhedron");
    export_UpdaterMuVT< ShapeSpheropolyhedron >(m, "UpdaterMuVTSpheropolyhedron");
    export_UpdaterClusters< ShapeSpheropolyhedron >(m, "UpdaterClustersSpheropolyhedron");
    export_UpdaterClustersImplicit< ShapeSpheropolyhedron, IntegratorHPMCMonoImplicit<ShapeSpheropolyhedron> >(m, "UpdaterClustersImplicitSpheropolyhedron");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron, IntegratorHPMCMonoImplicit<ShapeSpheropolyhedron> >(m, "UpdaterMuVTImplicitSpheropolyhedron");

    export_ExternalFieldInterface<ShapeSpheropolyhedron >(m, "ExternalFieldSpheropolyhedron");
    export_LatticeField<ShapeSpheropolyhedron >(m, "ExternalFieldLatticeSpheropolyhedron");
    export_ExternalFieldComposite<ShapeSpheropolyhedron >(m, "ExternalFieldCompositeSpheropolyhedron");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron >(m, "RemoveDriftUpdaterSpheropolyhedron");
    export_ExternalFieldWall<ShapeSpheropolyhedron >(m, "WallSpheropolyhedron");
    export_UpdaterExternalFieldWall<ShapeSpheropolyhedron >(m, "UpdaterExternalFieldWallSpheropolyhedron");
    export_ExternalCallback<ShapeSpheropolyhedron>(m, "ExternalCallbackSpheropolyhedron");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoGPUSpheropolyhedron");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoImplicitGPUSpheropolyhedron");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoImplicitNewGPUSpheropolyhedron");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeGPUSpheropolyhedron");

    #endif
    }

}
