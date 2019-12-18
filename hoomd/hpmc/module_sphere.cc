// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
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
#include "UpdaterClusters.h"

#ifdef ENABLE_HIP
#include "IntegratorHPMCMonoGPU.h"
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
    export_ComputeFreeVolume< ShapeSphere >(m, "ComputeFreeVolumeSphere");
    export_AnalyzerSDF< ShapeSphere >(m, "AnalyzerSDFSphere");
    export_UpdaterMuVT< ShapeSphere >(m, "UpdaterMuVTSphere");
    export_UpdaterClusters< ShapeSphere >(m, "UpdaterClustersSphere");

    export_ExternalFieldInterface<ShapeSphere>(m, "ExternalFieldSphere");
    export_LatticeField<ShapeSphere>(m, "ExternalFieldLatticeSphere");
    export_ExternalFieldComposite<ShapeSphere>(m, "ExternalFieldCompositeSphere");
    export_RemoveDriftUpdater<ShapeSphere>(m, "RemoveDriftUpdaterSphere");
    export_ExternalFieldWall<ShapeSphere>(m, "WallSphere");
    export_UpdaterExternalFieldWall<ShapeSphere>(m, "UpdaterExternalFieldWallSphere");
    export_ExternalCallback<ShapeSphere>(m, "ExternalCallbackSphere");

    #ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU< ShapeSphere >(m, "IntegratorHPMCMonoGPUSphere");
    export_ComputeFreeVolumeGPU< ShapeSphere >(m, "ComputeFreeVolumeGPUSphere");
    #endif
    }

}
