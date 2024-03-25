// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ShapeFacetedEllipsoid.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterClustersGPU.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Export the base HPMCMono integrators
void export_union_faceted_ellipsoid(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "IntegratorHPMCMonoFacetedEllipsoidUnion");
    export_ComputeFreeVolume<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "ComputeFreeVolumeFacetedEllipsoidUnion");
    export_ComputeSDF<ShapeUnion<ShapeFacetedEllipsoid>>(m, "ComputeSDFFacetedEllipsoidUnion");
    export_UpdaterMuVT<ShapeUnion<ShapeFacetedEllipsoid>>(m, "UpdaterMuVTFacetedEllipsoidUnion");
    export_UpdaterClusters<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "UpdaterClustersFacetedEllipsoidUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "ExternalFieldFacetedEllipsoidUnion");
    export_HarmonicField<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "ExternalFieldHarmonicFacetedEllipsoidUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeFacetedEllipsoid>>(m, "WallFacetedEllipsoidUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "IntegratorHPMCMonoFacetedEllipsoidUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "ComputeFreeVolumeFacetedEllipsoidUnionGPU");
    export_UpdaterClustersGPU<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "UpdaterClustersFacetedEllipsoidUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
