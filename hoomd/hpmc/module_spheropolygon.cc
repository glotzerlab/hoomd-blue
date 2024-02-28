// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSpheropolygon.h"
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
void export_spheropolygon(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSpheropolygon>(m, "IntegratorHPMCMonoSpheropolygon");
    export_ComputeFreeVolume<ShapeSpheropolygon>(m, "ComputeFreeVolumeSpheropolygon");
    export_ComputeSDF<ShapeSpheropolygon>(m, "ComputeSDFConvexSpheropolygon");
    export_UpdaterMuVT<ShapeSpheropolygon>(m, "UpdaterMuVTConvexSpheropolygon");
    export_UpdaterClusters<ShapeSpheropolygon>(m, "UpdaterClustersConvexSpheropolygon");

    export_ExternalFieldInterface<ShapeSpheropolygon>(m, "ExternalFieldSpheropolygon");
    export_HarmonicField<ShapeSpheropolygon>(m, "ExternalFieldHarmonicSpheropolygon");
    export_ExternalFieldWall<ShapeSpheropolygon>(m, "WallConvexSpheropolygon");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeSpheropolygon>(m, "IntegratorHPMCMonoSpheropolygonGPU");
    export_ComputeFreeVolumeGPU<ShapeSpheropolygon>(m, "ComputeFreeVolumeSpheropolygonGPU");
    export_UpdaterClustersGPU<ShapeSpheropolygon>(m, "UpdaterClustersConvexSpheropolygonGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
