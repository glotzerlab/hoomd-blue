// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSpheropolygon.h"

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
void export_spheropolygon(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSpheropolygon>(m, "IntegratorHPMCMonoSpheropolygon");
    export_ComputeFreeVolume<ShapeSpheropolygon>(m, "ComputeFreeVolumeSpheropolygon");
    export_ComputeSDF<ShapeSpheropolygon>(m, "ComputeSDFConvexSpheropolygon");
    export_UpdaterMuVT<ShapeSpheropolygon>(m, "UpdaterMuVTConvexSpheropolygon");
    export_UpdaterGCA<ShapeSpheropolygon>(m, "UpdaterGCAConvexSpheropolygon");

    export_ExternalFieldWall<ShapeSpheropolygon>(m, "WallConvexSpheropolygon");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeSpheropolygon>(m, "IntegratorHPMCMonoSpheropolygonGPU");
    export_ComputeFreeVolumeGPU<ShapeSpheropolygon>(m, "ComputeFreeVolumeSpheropolygonGPU");
    export_UpdaterGCAGPU<ShapeSpheropolygon>(m, "UpdaterGCAConvexSpheropolygonGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
