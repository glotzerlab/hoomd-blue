// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"

#include "ShapeConvexPolyhedron.h"
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
void export_convex_polyhedron(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoConvexPolyhedron");
    export_ComputeFreeVolume< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeConvexPolyhedron");
    export_AnalyzerSDF< ShapeConvexPolyhedron >(m, "AnalyzerSDFConvexPolyhedron");
    export_UpdaterMuVT< ShapeConvexPolyhedron >(m, "UpdaterMuVTConvexPolyhedron");
    export_UpdaterClusters< ShapeConvexPolyhedron >(m, "UpdaterClustersConvexPolyhedron");

    export_ExternalFieldInterface<ShapeConvexPolyhedron >(m, "ExternalFieldConvexPolyhedron");
    export_LatticeField<ShapeConvexPolyhedron >(m, "ExternalFieldLatticeConvexPolyhedron");
    export_ExternalFieldComposite<ShapeConvexPolyhedron >(m, "ExternalFieldCompositeConvexPolyhedron");
    export_RemoveDriftUpdater<ShapeConvexPolyhedron >(m, "RemoveDriftUpdaterConvexPolyhedron");
    export_ExternalFieldWall<ShapeConvexPolyhedron >(m, "WallConvexPolyhedron");
    export_UpdaterExternalFieldWall<ShapeConvexPolyhedron >(m, "UpdaterExternalFieldWallConvexPolyhedron");
    export_ExternalCallback<ShapeConvexPolyhedron>(m, "ExternalCallbackConvexPolyhedron");

    #ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU< ShapeConvexPolyhedron >(m, "IntegratorHPMCMonoGPUConvexPolyhedron");
    export_ComputeFreeVolumeGPU< ShapeConvexPolyhedron >(m, "ComputeFreeVolumeGPUConvexPolyhedron");

    #endif
    }

}
