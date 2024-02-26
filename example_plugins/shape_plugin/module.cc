// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "hoomd/hpmc/ComputeFreeVolume.h"
#include "hoomd/hpmc/IntegratorHPMC.h"
#include "hoomd/hpmc/IntegratorHPMCMono.h"

#include "hoomd/hpmc/ComputeSDF.h"
#include "hoomd/hpmc/ShapeUnion.h"

#include "hoomd/hpmc/ExternalField.h"
#include "hoomd/hpmc/ExternalFieldHarmonic.h"
#include "hoomd/hpmc/ExternalFieldWall.h"

#include "hoomd/hpmc/UpdaterClusters.h"
#include "hoomd/hpmc/UpdaterMuVT.h"

#include "ShapeMySphere.h"

#ifdef ENABLE_HIP
#include "hoomd/hpmc/ComputeFreeVolumeGPU.h"
#include "hoomd/hpmc/IntegratorHPMCMonoGPU.h"
#include "hoomd/hpmc/UpdaterClustersGPU.h"
#endif

using namespace hoomd::hpmc::detail;

namespace hoomd
    {
namespace hpmc
    {
//! Export the base HPMCMono integrators
PYBIND11_MODULE(_shape_plugin, m)
    {
    export_IntegratorHPMCMono<ShapeMySphere>(m, "IntegratorHPMCMonoMySphere");
    export_ComputeFreeVolume<ShapeMySphere>(m, "ComputeFreeVolumeMySphere");
    export_ComputeSDF<ShapeMySphere>(m, "ComputeSDFMySphere");
    export_UpdaterMuVT<ShapeMySphere>(m, "UpdaterMuVTMySphere");
    export_UpdaterClusters<ShapeMySphere>(m, "UpdaterClustersMySphere");

    export_ExternalFieldInterface<ShapeMySphere>(m, "ExternalFieldMySphere");
    export_HarmonicField<ShapeMySphere>(m, "ExternalFieldHarmonicMySphere");
    export_ExternalFieldWall<ShapeMySphere>(m, "WallMySphere");

    pybind11::class_<MySphereParams, std::shared_ptr<MySphereParams>>(m, "MySphereParams")
        .def(pybind11::init<pybind11::dict>())
        .def("asDict", &MySphereParams::asDict);

#ifdef ENABLE_HIP
    export_IntegratorHPMCMonoGPU<ShapeMySphere>(m, "IntegratorHPMCMonoMySphereGPU");
    export_ComputeFreeVolumeGPU<ShapeMySphere>(m, "ComputeFreeVolumeMySphereGPU");
    export_UpdaterClustersGPU<ShapeMySphere>(m, "UpdaterClustersMySphereGPU");
#endif
    }

    } // namespace hpmc
    } // namespace hoomd
