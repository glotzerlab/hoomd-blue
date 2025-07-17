// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSimplePolygon.h"

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
void export_simple_polygon(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSimplePolygon>(m, "IntegratorHPMCMonoSimplePolygon");
    export_ComputeFreeVolume<ShapeSimplePolygon>(m, "ComputeFreeVolumeSimplePolygon");
    export_ComputeSDF<ShapeSimplePolygon>(m, "ComputeSDFSimplePolygon");
    export_UpdaterMuVT<ShapeSimplePolygon>(m, "UpdaterMuVTSimplePolygon");
    export_UpdaterGCA<ShapeSimplePolygon>(m, "UpdaterGCASimplePolygon");

    export_ExternalFieldWall<ShapeSimplePolygon>(m, "WallSimplePolygon");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeSimplePolygon>(m, "IntegratorHPMCMonoSimplePolygonGPU");
    export_ComputeFreeVolumeGPU<ShapeSimplePolygon>(m, "ComputeFreeVolumeSimplePolygonGPU");
    export_UpdaterGCAGPU<ShapeSimplePolygon>(m, "UpdaterGCASimplePolygonGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
