// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ShapeSpheropolyhedron.h"
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
void export_union_convex_polyhedron(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "IntegratorHPMCMonoConvexPolyhedronUnion");
    export_ComputeFreeVolume<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ComputeFreeVolumeConvexPolyhedronUnion");
    export_ComputeSDF<ShapeUnion<ShapeSpheropolyhedron>>(m,
                                                         "ComputeSDFConvexSpheropolyhedronUnion");
    export_UpdaterMuVT<ShapeUnion<ShapeSpheropolyhedron>>(m,
                                                          "UpdaterMuVTConvexSpheropolyhedronUnion");
    export_UpdaterClusters<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "UpdaterClustersConvexSpheropolyhedronUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ExternalFieldConvexPolyhedronUnion");
    export_HarmonicField<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ExternalFieldHarmonicConvexPolyhedronUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSpheropolyhedron>>(m,
                                                                "WallConvexSpheropolyhedronUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "IntegratorHPMCMonoConvexPolyhedronUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ComputeFreeVolumeConvexPolyhedronUnionGPU");
    export_UpdaterClustersGPU<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "UpdaterClustersConvexSpheropolyhedronUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
