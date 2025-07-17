// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapePolyhedron.h"

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
void export_polyhedron(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapePolyhedron>(m, "IntegratorHPMCMonoPolyhedron");
    export_ComputeFreeVolume<ShapePolyhedron>(m, "ComputeFreeVolumePolyhedron");
    export_ComputeSDF<ShapePolyhedron>(m, "ComputeSDFPolyhedron");
    export_UpdaterMuVT<ShapePolyhedron>(m, "UpdaterMuVTPolyhedron");
    export_UpdaterGCA<ShapePolyhedron>(m, "UpdaterGCAPolyhedron");

    export_ExternalFieldWall<ShapePolyhedron>(m, "WallPolyhedron");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapePolyhedron>(m, "IntegratorHPMCMonoPolyhedronGPU");
    export_ComputeFreeVolumeGPU<ShapePolyhedron>(m, "ComputeFreeVolumePolyhedronGPU");
    export_UpdaterGCAGPU<ShapePolyhedron>(m, "UpdaterGCAPolyhedronGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
