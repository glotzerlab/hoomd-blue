// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMCMono.h"

#include "ShapeFacetedEllipsoid.h"
#include "ShapeUnion.h"

#include "ExternalFieldWall.h"

#include "UpdaterGCA.h"
#include "UpdaterMuVT.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterGCAGPU.h"
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
    export_UpdaterGCA<ShapeUnion<ShapeFacetedEllipsoid>>(m, "UpdaterGCAFacetedEllipsoidUnion");

    export_ExternalFieldWall<ShapeUnion<ShapeFacetedEllipsoid>>(m, "WallFacetedEllipsoidUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "IntegratorHPMCMonoFacetedEllipsoidUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeFacetedEllipsoid>>(
        m,
        "ComputeFreeVolumeFacetedEllipsoidUnionGPU");
    export_UpdaterGCAGPU<ShapeUnion<ShapeFacetedEllipsoid>>(m,
                                                            "UpdaterGCAFacetedEllipsoidUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
