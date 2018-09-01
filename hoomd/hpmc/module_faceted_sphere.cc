// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeFacetedSphere.h"
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
void export_faceted_sphere(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeFacetedSphere >(m, "IntegratorHPMCMonoFacetedSphere");
    export_IntegratorHPMCMonoImplicit< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitFacetedSphere");
    export_ComputeFreeVolume< ShapeFacetedSphere >(m, "ComputeFreeVolumeFacetedSphere");
    export_AnalyzerSDF< ShapeFacetedSphere >(m, "AnalyzerSDFFacetedSphere");
    export_UpdaterMuVT< ShapeFacetedSphere >(m, "UpdaterMuVTFacetedSphere");
    export_UpdaterClusters< ShapeFacetedSphere >(m, "UpdaterClustersFacetedSphere");
    export_UpdaterClustersImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterClustersImplicitFacetedSphere");
    export_UpdaterMuVTImplicit< ShapeFacetedSphere, IntegratorHPMCMonoImplicit<ShapeFacetedSphere> >(m, "UpdaterMuVTImplicitFacetedSphere");

    export_ExternalFieldInterface<ShapeFacetedSphere>(m, "ExternalFieldFacetedSphere");
    export_LatticeField<ShapeFacetedSphere>(m, "ExternalFieldLatticeFacetedSphere");
    export_ExternalFieldComposite<ShapeFacetedSphere>(m, "ExternalFieldCompositeFacetedSphere");
    export_RemoveDriftUpdater<ShapeFacetedSphere>(m, "RemoveDriftUpdaterFacetedSphere");
    export_ExternalFieldWall<ShapeFacetedSphere>(m, "WallFacetedSphere");
    export_UpdaterExternalFieldWall<ShapeFacetedSphere>(m, "UpdaterExternalFieldWallFacetedSphere");
    export_ExternalCallback<ShapeFacetedSphere>(m, "ExternalCallbackFacetedSphere");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoGPUFacetedSphere");
    export_IntegratorHPMCMonoImplicitGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitGPUFacetedSphere");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeFacetedSphere >(m, "IntegratorHPMCMonoImplicitNewGPUFacetedSphere");
    export_ComputeFreeVolumeGPU< ShapeFacetedSphere >(m, "ComputeFreeVolumeGPUFacetedSphere");
    #endif
    }

}
