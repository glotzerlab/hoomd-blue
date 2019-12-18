// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "ComputeFreeVolume.h"

#include "ShapeSimplePolygon.h"
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
void export_simple_polygon(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSimplePolygon >(m, "IntegratorHPMCMonoSimplePolygon");
    export_ComputeFreeVolume< ShapeSimplePolygon >(m, "ComputeFreeVolumeSimplePolygon");
    export_AnalyzerSDF< ShapeSimplePolygon >(m, "AnalyzerSDFSimplePolygon");
    export_UpdaterMuVT< ShapeSimplePolygon >(m, "UpdaterMuVTSimplePolygon");
    export_UpdaterClusters< ShapeSimplePolygon >(m, "UpdaterClustersSimplePolygon");

    export_ExternalFieldInterface<ShapeSimplePolygon>(m, "ExternalFieldSimplePolygon");
    export_LatticeField<ShapeSimplePolygon>(m, "ExternalFieldLatticeSimplePolygon");
    export_ExternalFieldComposite<ShapeSimplePolygon>(m, "ExternalFieldCompositeSimplePolygon");
    export_RemoveDriftUpdater<ShapeSimplePolygon>(m, "RemoveDriftUpdaterSimplePolygon");
    // export_ExternalFieldWall<ShapeSimplePolygon>(m, "WallSimplePolygon");
    // export_UpdaterExternalFieldWall<ShapeSimplePolygon>(m, "UpdaterExternalFieldWallSimplePolygon");
    export_ExternalCallback<ShapeSimplePolygon>(m, "ExternalCallbackSimplePolygon");

    #ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU< ShapeSimplePolygon >(m, "IntegratorHPMCMonoGPUSimplePolygon");
    export_ComputeFreeVolumeGPU< ShapeSimplePolygon >(m, "ComputeFreeVolumeGPUSimplePolygon");
    #endif
    }

}
