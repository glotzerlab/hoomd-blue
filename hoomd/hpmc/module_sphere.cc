// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSphere.h"
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
void export_sphere(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSphere >(m, "IntegratorHPMCMonoSphere");
    export_IntegratorHPMCMonoImplicit< ShapeSphere >(m, "IntegratorHPMCMonoImplicitSphere");
    export_ComputeFreeVolume< ShapeSphere >(m, "ComputeFreeVolumeSphere");
    export_AnalyzerSDF< ShapeSphere >(m, "AnalyzerSDFSphere");
    export_UpdaterMuVT< ShapeSphere >(m, "UpdaterMuVTSphere");
    export_UpdaterClusters< ShapeSphere >(m, "UpdaterClustersSphere");
    export_UpdaterClustersImplicit< ShapeSphere,IntegratorHPMCMonoImplicit<ShapeSphere> >(m, "UpdaterClustersImplicitSphere");
    export_UpdaterMuVTImplicit< ShapeSphere, IntegratorHPMCMonoImplicit<ShapeSphere> >(m, "UpdaterMuVTImplicitSphere");

    export_ExternalFieldInterface<ShapeSphere>(m, "ExternalFieldSphere");
    export_LatticeField<ShapeSphere>(m, "ExternalFieldLatticeSphere");
    export_ExternalFieldComposite<ShapeSphere>(m, "ExternalFieldCompositeSphere");
    export_RemoveDriftUpdater<ShapeSphere>(m, "RemoveDriftUpdaterSphere");
    export_ExternalFieldWall<ShapeSphere>(m, "WallSphere");
    export_UpdaterExternalFieldWall<ShapeSphere>(m, "UpdaterExternalFieldWallSphere");
    export_ExternalCallback<ShapeSphere>(m, "ExternalCallbackSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeSphere >(m, "IntegratorHPMCMonoGPUSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSphere >(m, "IntegratorHPMCMonoImplicitGPUSphere");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeSphere >(m, "IntegratorHPMCMonoImplicitNewGPUSphere");
    export_ComputeFreeVolumeGPU< ShapeSphere >(m, "ComputeFreeVolumeGPUSphere");
    #endif
    }

}
