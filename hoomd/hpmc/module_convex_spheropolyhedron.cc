// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolyhedron.h"
#include "ComputeSDF.h"
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
#include "UpdaterClustersGPU.h"
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
    export_ComputeFreeVolume< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeSpheropolyhedron");
    export_ComputeSDF< ShapeSpheropolyhedron >(m, "ComputeSDFSpheropolyhedron");
    export_UpdaterMuVT< ShapeSpheropolyhedron >(m, "UpdaterMuVTConvexSpheropolyhedron");
    export_UpdaterClusters< ShapeSpheropolyhedron >(m, "UpdaterClustersConvexSpheropolyhedron");

    export_ExternalFieldInterface<ShapeSpheropolyhedron >(m, "ExternalFieldSpheropolyhedron");
    export_LatticeField<ShapeSpheropolyhedron >(m, "ExternalFieldLatticeSpheropolyhedron");
    export_ExternalFieldComposite<ShapeSpheropolyhedron >(m, "ExternalFieldCompositeSpheropolyhedron");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron >(m, "RemoveDriftUpdaterSpheropolyhedron");
    export_ExternalFieldWall<ShapeSpheropolyhedron >(m, "WallSpheropolyhedron");
    export_UpdaterExternalFieldWall<ShapeSpheropolyhedron >(m, "UpdaterExternalFieldWallSpheropolyhedron");
    export_ExternalCallback<ShapeSpheropolyhedron>(m, "ExternalCallbackSpheropolyhedron");

    #ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron >(m, "IntegratorHPMCMonoSpheropolyhedronGPU");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron >(m, "ComputeFreeVolumeSpheropolyhedronGPU");
    export_UpdaterClustersGPU< ShapeSpheropolyhedron >(m, "UpdaterClustersConvexSpheropolyhedronGPU");

    #endif
    }

}
