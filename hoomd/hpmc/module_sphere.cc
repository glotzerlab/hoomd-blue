// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoNEC.h"

#include "ComputeSDF.h"
#include "ShapeSphere.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldHarmonic.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterMuVT.h"
#include "UpdaterVirtualMoveMonteCarlo.h"

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
void export_sphere(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSphere>(m, "IntegratorHPMCMonoSphere");
    export_IntegratorHPMCMonoNEC<ShapeSphere>(m, "IntegratorHPMCMonoNECSphere");
    export_ComputeFreeVolume<ShapeSphere>(m, "ComputeFreeVolumeSphere");
    export_ComputeSDF<ShapeSphere>(m, "ComputeSDFSphere");
    export_UpdaterMuVT<ShapeSphere>(m, "UpdaterMuVTSphere");
    export_UpdaterClusters<ShapeSphere>(m, "UpdaterClustersSphere");
    export_UpdaterVirtualMoveMonteCarlo<ShapeSphere>(m, "UpdaterVMMCSphere");

    export_ExternalFieldInterface<ShapeSphere>(m, "ExternalFieldSphere");
    export_HarmonicField<ShapeSphere>(m, "ExternalFieldHarmonicSphere");
    export_ExternalFieldWall<ShapeSphere>(m, "WallSphere");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeSphere>(m, "IntegratorHPMCMonoSphereGPU");
    export_ComputeFreeVolumeGPU<ShapeSphere>(m, "ComputeFreeVolumeSphereGPU");
    export_UpdaterClustersGPU<ShapeSphere>(m, "UpdaterClustersSphereGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
