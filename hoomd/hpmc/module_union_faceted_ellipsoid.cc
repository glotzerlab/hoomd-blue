// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"
#include "AnalyzerSDF.h"

#include "ShapeUnion.h"
#include "ShapeFacetedEllipsoid.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

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
void export_union_faceted_ellipsoid(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeFacetedEllipsoid> >(m, "IntegratorHPMCMonoFacetedEllipsoidUnion");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeFacetedEllipsoid> >(m, "IntegratorHPMCMonoImplicitFacetedEllipsoidUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeFacetedEllipsoid> >(m, "ComputeFreeVolumeFacetedEllipsoidUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeFacetedEllipsoid> >(m, "AnalyzerSDFFacetedEllipsoidUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeFacetedEllipsoid> >(m, "UpdaterMuVTFacetedEllipsoidUnion");
    export_UpdaterClusters<ShapeUnion<ShapeFacetedEllipsoid> >(m, "UpdaterClustersFacetedEllipsoidUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeFacetedEllipsoid>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeFacetedEllipsoid> > >(m, "UpdaterClustersImplicitFacetedEllipsoidUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeFacetedEllipsoid>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeFacetedEllipsoid> > >(m, "UpdaterMuVTImplicitFacetedEllipsoidUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeFacetedEllipsoid> >(m, "ExternalFieldFacetedEllipsoidUnion");
    export_LatticeField<ShapeUnion<ShapeFacetedEllipsoid> >(m, "ExternalFieldLatticeFacetedEllipsoidUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeFacetedEllipsoid> >(m, "ExternalFieldCompositeFacetedEllipsoidUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeFacetedEllipsoid> >(m, "RemoveDriftUpdaterFacetedEllipsoidUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeFacetedEllipsoid> >(m, "WallFacetedEllipsoidUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeFacetedEllipsoid> >(m, "UpdaterExternalFieldWallFacetedEllipsoidUnion");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeFacetedEllipsoid> >(m, "IntegratorHPMCMonoGPUFacetedEllipsoidUnion");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeFacetedEllipsoid> >(m, "IntegratorHPMCMonoImplicitGPUFacetedEllipsoidUnion");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeUnion<ShapeFacetedEllipsoid> >(m, "IntegratorHPMCMonoImplicitNewGPUFacetedEllipsoidUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeFacetedEllipsoid> >(m, "ComputeFreeVolumeGPUFacetedEllipsoidUnion");

    #endif
    }

}
