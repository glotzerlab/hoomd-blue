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
void export_convex_spheropolyhedron32(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<32> >(m, "IntegratorHPMCMonoSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<32> >(m, "IntegratorHPMCMonoImplicitSpheropolyhedron32");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<32> >(m, "ComputeFreeVolumeSpheropolyhedron32");
    export_AnalyzerSDF< ShapeSpheropolyhedron<32> >(m, "AnalyzerSDFSpheropolyhedron32");
    export_UpdaterMuVT< ShapeSpheropolyhedron<32> >(m, "UpdaterMuVTSpheropolyhedron32");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<32> >(m, "UpdaterMuVTImplicitSpheropolyhedron32");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<32> >(m, "ExternalFieldSpheropolyhedron32");
    export_LatticeField<ShapeSpheropolyhedron<32> >(m, "ExternalFieldLatticeSpheropolyhedron32");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<32> >(m, "ExternalFieldCompositeSpheropolyhedron32");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<32> >(m, "RemoveDriftUpdaterSpheropolyhedron32");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<32> >(m, "WallSpheropolyhedron32");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<32> >(m, "UpdaterExternalFieldWallSpheropolyhedron32");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<32> >(m, "IntegratorHPMCMonoGPUSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<32> >(m, "IntegratorHPMCMonoImplicitGPUSpheropolyhedron32");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<32> >(m, "ComputeFreeVolumeGPUSpheropolyhedron32");

    #endif
    }

}
