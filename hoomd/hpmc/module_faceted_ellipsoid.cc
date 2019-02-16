// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeFacetedEllipsoid.h"
#include "AnalyzerSDF.h"

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
void export_faceted_ellipsoid(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeFacetedEllipsoid >(m, "IntegratorHPMCMonoFacetedEllipsoid");
    export_IntegratorHPMCMonoImplicit< ShapeFacetedEllipsoid >(m, "IntegratorHPMCMonoImplicitFacetedEllipsoid");
    export_ComputeFreeVolume< ShapeFacetedEllipsoid >(m, "ComputeFreeVolumeFacetedEllipsoid");
    export_AnalyzerSDF< ShapeFacetedEllipsoid >(m, "AnalyzerSDFFacetedEllipsoid");
    export_UpdaterMuVT< ShapeFacetedEllipsoid >(m, "UpdaterMuVTFacetedEllipsoid");
    export_UpdaterClusters< ShapeFacetedEllipsoid >(m, "UpdaterClustersFacetedEllipsoid");
    export_UpdaterClustersImplicit< ShapeFacetedEllipsoid, IntegratorHPMCMonoImplicit<ShapeFacetedEllipsoid> >(m, "UpdaterClustersImplicitFacetedEllipsoid");
    export_UpdaterMuVTImplicit< ShapeFacetedEllipsoid, IntegratorHPMCMonoImplicit<ShapeFacetedEllipsoid> >(m, "UpdaterMuVTImplicitFacetedEllipsoid");

    export_ExternalFieldInterface<ShapeFacetedEllipsoid>(m, "ExternalFieldFacetedEllipsoid");
    export_LatticeField<ShapeFacetedEllipsoid>(m, "ExternalFieldLatticeFacetedEllipsoid");
    export_ExternalFieldComposite<ShapeFacetedEllipsoid>(m, "ExternalFieldCompositeFacetedEllipsoid");
    export_RemoveDriftUpdater<ShapeFacetedEllipsoid>(m, "RemoveDriftUpdaterFacetedEllipsoid");
    export_ExternalFieldWall<ShapeFacetedEllipsoid>(m, "WallFacetedEllipsoid");
    export_UpdaterExternalFieldWall<ShapeFacetedEllipsoid>(m, "UpdaterExternalFieldWallFacetedEllipsoid");
    export_ExternalCallback<ShapeFacetedEllipsoid>(m, "ExternalCallbackFacetedEllipsoid");

    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeFacetedEllipsoid >(m, "IntegratorHPMCMonoGPUFacetedEllipsoid");
    export_IntegratorHPMCMonoImplicitGPU< ShapeFacetedEllipsoid >(m, "IntegratorHPMCMonoImplicitGPUFacetedEllipsoid");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeFacetedEllipsoid >(m, "IntegratorHPMCMonoImplicitNewGPUFacetedEllipsoid");
    export_ComputeFreeVolumeGPU< ShapeFacetedEllipsoid >(m, "ComputeFreeVolumeGPUFacetedEllipsoid");
    #endif
    }

}
