// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "AnalyzerSDF.h"
#include "ShapeEllipsoid.h"
#include "ShapeUnion.h"

#include "ExternalCallback.h"
#include "ExternalField.h"
#include "ExternalFieldComposite.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldWall.h"

#include "UpdaterClusters.h"
#include "UpdaterExternalFieldWall.h"
#include "UpdaterMuVT.h"
#include "UpdaterRemoveDrift.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterClustersGPU.h"
#endif

namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
    {
//! Export the base HPMCMono integrators
void export_ellipsoid(py::module& m)
    {
    export_IntegratorHPMCMono<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoid");
    export_ComputeFreeVolume<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoid");
    export_AnalyzerSDF<ShapeEllipsoid>(m, "AnalyzerSDFEllipsoid");
    export_UpdaterMuVT<ShapeEllipsoid>(m, "UpdaterMuVTEllipsoid");
    export_UpdaterClusters<ShapeEllipsoid>(m, "UpdaterClustersEllipsoid");

    export_ExternalFieldInterface<ShapeEllipsoid>(m, "ExternalFieldEllipsoid");
    export_LatticeField<ShapeEllipsoid>(m, "ExternalFieldLatticeEllipsoid");
    export_ExternalFieldComposite<ShapeEllipsoid>(m, "ExternalFieldCompositeEllipsoid");
    export_RemoveDriftUpdater<ShapeEllipsoid>(m, "RemoveDriftUpdaterEllipsoid");
    export_ExternalFieldWall<ShapeEllipsoid>(m, "WallEllipsoid");
    export_UpdaterExternalFieldWall<ShapeEllipsoid>(m, "UpdaterExternalFieldWallEllipsoid");
    export_ExternalCallback<ShapeEllipsoid>(m, "ExternalCallbackEllipsoid");

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeEllipsoid>(m, "IntegratorHPMCMonoEllipsoidGPU");
    export_ComputeFreeVolumeGPU<ShapeEllipsoid>(m, "ComputeFreeVolumeEllipsoidGPU");
    export_UpdaterClustersGPU<ShapeEllipsoid>(m, "UpdaterClustersEllipsoidGPU");
#endif
    }

    } // namespace hpmc
