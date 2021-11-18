// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ShapeSpheropolyhedron.h"
#include "ShapeUnion.h"

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
    export_LatticeField<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ExternalFieldLatticeConvexPolyhedronUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "ExternalFieldCompositeConvexPolyhedronUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSpheropolyhedron>>(m, "WallConvexPolyhedronUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSpheropolyhedron>>(
        m,
        "UpdaterExternalFieldWallConvexPolyhedronUnion");

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
