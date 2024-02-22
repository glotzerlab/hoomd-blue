// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

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
void export_union_sphere(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnion");
    export_ComputeFreeVolume<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnion");
    export_ComputeSDF<ShapeUnion<ShapeSphere>>(m, "ComputeSDFSphereUnion");
    export_UpdaterMuVT<ShapeUnion<ShapeSphere>>(m, "UpdaterMuVTSphereUnion");
    export_UpdaterClusters<ShapeUnion<ShapeSphere>>(m, "UpdaterClustersSphereUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere>>(m, "ExternalFieldSphereUnion");
    export_HarmonicField<ShapeUnion<ShapeSphere>>(m, "ExternalFieldHarmonicSphereUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere>>(m, "WallSphereUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnionGPU");
    export_UpdaterClustersGPU<ShapeUnion<ShapeSphere>>(m, "UpdaterClustersSphereUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
