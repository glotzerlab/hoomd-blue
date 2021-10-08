// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "ComputeSDF.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

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
void export_union_sphere(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnion");
    export_ComputeFreeVolume<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnion");
    export_ComputeSDF<ShapeUnion<ShapeSphere>>(m, "ComputeSDFSphereUnion");
    export_UpdaterMuVT<ShapeUnion<ShapeSphere>>(m, "UpdaterMuVTSphereUnion");
    export_UpdaterClusters<ShapeUnion<ShapeSphere>>(m, "UpdaterClustersSphereUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphere>>(m, "ExternalFieldSphereUnion");
    export_LatticeField<ShapeUnion<ShapeSphere>>(m, "ExternalFieldLatticeSphereUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphere>>(m, "ExternalFieldCompositeSphereUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSphere>>(m, "WallSphereUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphere>>(m,
                                                             "UpdaterExternalFieldWallSphereUnion");
    export_ExternalCallback<ShapeUnion<ShapeSphere>>(m, "ExternalCallbackSphereUnion");

#ifdef ENABLE_HIP

    export_IntegratorHPMCMonoGPU<ShapeUnion<ShapeSphere>>(m, "IntegratorHPMCMonoSphereUnionGPU");
    export_ComputeFreeVolumeGPU<ShapeUnion<ShapeSphere>>(m, "ComputeFreeVolumeSphereUnionGPU");
    export_UpdaterClustersGPU<ShapeUnion<ShapeSphere>>(m, "UpdaterClustersSphereUnionGPU");

#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
