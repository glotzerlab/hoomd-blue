// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"

#include "ShapeSpheropolyhedron.h"
#include "ShapeSphinx.h"
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
void export_convex_spheropolyhedron64(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeSpheropolyhedron<64> >(m, "IntegratorHPMCMonoSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicit< ShapeSpheropolyhedron<64> >(m, "IntegratorHPMCMonoImplicitSpheropolyhedron64");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<64> >(m, "ComputeFreeVolumeSpheropolyhedron64");
    export_AnalyzerSDF< ShapeSpheropolyhedron<64> >(m, "AnalyzerSDFSpheropolyhedron64");
    export_UpdaterMuVT< ShapeSpheropolyhedron<64> >(m, "UpdaterMuVTSpheropolyhedron64");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<64> >(m, "UpdaterMuVTImplicitSpheropolyhedron64");

    export_ExternalFieldInterface<ShapeSpheropolyhedron<64> >(m, "ExternalFieldSpheropolyhedron64");
    export_LatticeField<ShapeSpheropolyhedron<64> >(m, "ExternalFieldLatticeSpheropolyhedron64");
    export_ExternalFieldComposite<ShapeSpheropolyhedron<64> >(m, "ExternalFieldCompositeSpheropolyhedron64");
    export_RemoveDriftUpdater<ShapeSpheropolyhedron<64> >(m, "RemoveDriftUpdaterSpheropolyhedron64");
    // export_ExternalFieldWall<ShapeSpheropolyhedron<64> >(m, "WallSpheropolyhedron64");
    // export_UpdaterExternalFieldWall<ShapeSpheropolyhedron<64> >(m, "UpdaterExternalFieldWallSpheropolyhedron64");

    #ifdef ENABLE_CUDA

    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<64> >(m, "IntegratorHPMCMonoGPUSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<64> >(m, "IntegratorHPMCMonoImplicitGPUSpheropolyhedron64");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<64> >(m, "ComputeFreeVolumeGPUSpheropolyhedron64");

    #endif
    }

}
