// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolyhedron.h"
#include "AnalyzerSDF.h"
#include "ShapeUnion.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif




namespace py = pybind11;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_convex_spheropolyhedron8(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<8> >(m, "IntegratorHPMCMonoSpheropolyhedron8");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<8> >(m, "IntegratorHPMCMonoImplicitSpheropolyhedron8");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<8> >(m, "ComputeFreeVolumeSpheropolyhedron8");
    export_AnalyzerSDF< ShapeSpheropolyhedron<8> >(m, "AnalyzerSDFSpheropolyhedron8");
    export_UpdaterMuVT< ShapeSpheropolyhedron<8> >(m, "UpdaterMuVTSpheropolyhedron8");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<8> >(m, "UpdaterMuVTImplicitSpheropolyhedron8");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<8> >(m, "ExternalFieldSpheropolyhedron8");
    export_LatticeField<ShapeSpheropolyhedron<8> >(m, "ExternalFieldLatticeSpheropolyhedron8");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<8> >(m, "ExternalFieldCompositeSpheropolyhedron8");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<8> >(m, "RemoveDriftUpdaterSpheropolyhedron8");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<8> >(m, "WallSpheropolyhedron8");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<8> >(m, "UpdaterExternalFieldWallSpheropolyhedron8");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<8> >(m, "IntegratorHPMCMonoGPUSpheropolyhedron8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<8> >(m, "IntegratorHPMCMonoImplicitGPUSpheropolyhedron8");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<8> >(m, "ComputeFreeVolumeGPUSpheropolyhedron8");

    #endif
    }

}
