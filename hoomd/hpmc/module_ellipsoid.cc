// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeEllipsoid.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
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
void export_ellipsoid(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoid");
    export_ComputeFreeVolume<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoid");
    export_ComputeSDF<ShapeEllipsoid>(m, "ComputeSDFEllipsoid");
    export_UpdaterMuVT<ShapeEllipsoid>(m, "UpdaterMuVTEllipsoid");
    export_UpdaterClusters<ShapeEllipsoid>(m, "UpdaterClustersEllipsoid");

    export_ExternalFieldInterface<ShapeEllipsoid>(m, "ExternalFieldEllipsoid");
    export_HarmonicField<ShapeEllipsoid>(m, "ExternalFieldHarmonicEllipsoid");
    export_ExternalFieldComposite<ShapeEllipsoid>(m, "ExternalFieldCompositeEllipsoid");
    export_ExternalFieldWall<ShapeEllipsoid>(m, "WallEllipsoid");
    export_ExternalCallback<ShapeEllipsoid>(m, "ExternalCallbackEllipsoid");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoidGPU");
    export_ComputeFreeVolumeGPU<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoidGPU");
    export_UpdaterClustersGPU<ShapeEllipsoid>(m, "UpdaterClustersEllipsoidGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
