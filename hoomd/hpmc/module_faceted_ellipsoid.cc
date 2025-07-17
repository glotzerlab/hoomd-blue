// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeFacetedEllipsoid.h"

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
void export_faceted_ellipsoid(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeFacetedEllipsoid>(m, "IntegratorHPMCMonoFacetedEllipsoid");
    export_ComputeFreeVolume<ShapeFacetedEllipsoid>(m, "ComputeFreeVolumeFacetedEllipsoid");
    export_ComputeSDF<ShapeFacetedEllipsoid>(m, "ComputeSDFFacetedEllipsoid");
    export_UpdaterMuVT<ShapeFacetedEllipsoid>(m, "UpdaterMuVTFacetedEllipsoid");
    export_UpdaterGCA<ShapeFacetedEllipsoid>(m, "UpdaterGCAFacetedEllipsoid");

    export_ExternalFieldWall<ShapeFacetedEllipsoid>(m, "WallFacetedEllipsoid");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeFacetedEllipsoid>(m, "IntegratorHPMCMonoFacetedEllipsoidGPU");
    export_ComputeFreeVolumeGPU<ShapeFacetedEllipsoid>(m, "ComputeFreeVolumeFacetedEllipsoidGPU");
    export_UpdaterGCAGPU<ShapeFacetedEllipsoid>(m, "UpdaterGCAFacetedEllipsoidGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
