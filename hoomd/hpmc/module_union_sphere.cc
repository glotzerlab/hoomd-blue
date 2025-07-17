// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMCMono.h"

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
void export_union_sphere(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnion");
    export_ComputeFreeVolume<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnion");
    export_ComputeSDF<ShapeUnion<ShapeSphere>>(m, "ComputeSDFSphereUnion");
    export_UpdaterMuVT<ShapeUnion<ShapeSphere>>(m, "UpdaterMuVTSphereUnion");
    export_UpdaterGCA<ShapeUnion<ShapeSphere>>(m, "UpdaterGCASphereUnion");

    export_ExternalFieldWall<ShapeUnion<ShapeSphere>>(m, "WallSphereUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnionGPU");
    export_UpdaterGCAGPU<ShapeUnion<ShapeSphere>>(m, "UpdaterGCASphereUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
