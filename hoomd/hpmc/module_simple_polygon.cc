// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSimplePolygon.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterExternalFieldWall.h"
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
void export_simple_polygon(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSimplePolygon>(m, "IntegratorHPMCMonoSimplePolygon");
    export_ComputeFreeVolume<ShapeSimplePolygon>(m, "ComputeFreeVolumeSimplePolygon");
    export_ComputeSDF<ShapeSimplePolygon>(m, "ComputeSDFSimplePolygon");
    export_UpdaterMuVT<ShapeSimplePolygon>(m, "UpdaterMuVTSimplePolygon");
    export_UpdaterClusters<ShapeSimplePolygon>(m, "UpdaterClustersSimplePolygon");

    export_ExternalFieldInterface<ShapeSimplePolygon>(m, "ExternalFieldSimplePolygon");
    export_LatticeField<ShapeSimplePolygon>(m, "ExternalFieldLatticeSimplePolygon");
    export_ExternalFieldComposite<ShapeSimplePolygon>(m, "ExternalFieldCompositeSimplePolygon");
    // export_ExternalFieldWall<ShapeSimplePolygon>(m, "WallSimplePolygon");
    // export_UpdaterExternalFieldWall<ShapeSimplePolygon>(m,
    // "UpdaterExternalFieldWallSimplePolygon");
    export_ExternalCallback<ShapeSimplePolygon>(m, "ExternalCallbackSimplePolygon");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeSimplePolygon>(m, "IntegratorHPMCMonoSimplePolygonGPU");
    export_ComputeFreeVolumeGPU<ShapeSimplePolygon>(m, "ComputeFreeVolumeSimplePolygonGPU");
    export_UpdaterClustersGPU<ShapeSimplePolygon>(m, "UpdaterClustersSimplePolygonGPU");
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
