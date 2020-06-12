// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeEllipsoid.h"
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
void export_ellipsoid(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeEllipsoid >(m, "IntegratorHPMCMonoEllipsoid");
    export_IntegratorHPMCMonoImplicit< ShapeEllipsoid >(m, "IntegratorHPMCMonoImplicitEllipsoid");
    export_ComputeFreeVolume< ShapeEllipsoid >(m, "ComputeFreeVolumeEllipsoid");
    export_AnalyzerSDF< ShapeEllipsoid >(m, "AnalyzerSDFEllipsoid");
    export_UpdaterMuVT< ShapeEllipsoid >(m, "UpdaterMuVTEllipsoid");
    export_UpdaterClusters< ShapeEllipsoid >(m, "UpdaterClustersEllipsoid");
    export_UpdaterClustersImplicit< ShapeEllipsoid, IntegratorHPMCMonoImplicit<ShapeEllipsoid> >(m, "UpdaterClustersImplicitEllipsoid");
    export_UpdaterMuVTImplicit< ShapeEllipsoid, IntegratorHPMCMonoImplicit<ShapeEllipsoid> >(m, "UpdaterMuVTImplicitEllipsoid");

    export_ExternalFieldInterface<ShapeEllipsoid>(m, "ExternalFieldEllipsoid");
    export_LatticeField<ShapeEllipsoid>(m, "ExternalFieldLatticeEllipsoid");
    export_ExternalFieldComposite<ShapeEllipsoid>(m, "ExternalFieldCompositeEllipsoid");
    export_RemoveDriftUpdater<ShapeEllipsoid>(m, "RemoveDriftUpdaterEllipsoid");
    export_ExternalFieldWall<ShapeEllipsoid>(m, "WallEllipsoid");
    export_UpdaterExternalFieldWall<ShapeEllipsoid>(m, "UpdaterExternalFieldWallEllipsoid");
    export_ExternalCallback<ShapeEllipsoid>(m, "ExternalCallbackEllipsoid");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeEllipsoid >(m, "IntegratorHPMCMonoGPUEllipsoid");
    export_IntegratorHPMCMonoImplicitGPU< ShapeEllipsoid >(m, "IntegratorHPMCMonoImplicitGPUEllipsoid");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeEllipsoid >(m, "IntegratorHPMCMonoImplicitNewGPUEllipsoid");
    export_ComputeFreeVolumeGPU< ShapeEllipsoid >(m, "ComputeFreeVolumeGPUEllipsoid");
    #endif
    }

}
