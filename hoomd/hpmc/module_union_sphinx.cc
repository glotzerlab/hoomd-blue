// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMonoImplicit.h"
#include "ComputeFreeVolume.h"
#include "AnalyzerSDF.h"

#include "ShapeUnion.h"
#include "ShapeSphinx.h"

#include "ExternalField.h"
#include "ExternalFieldWall.h"
#include "ExternalFieldLattice.h"
#include "ExternalFieldComposite.h"

#include "UpdaterExternalFieldWall.h"
#include "UpdaterRemoveDrift.h"
#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"
#include "UpdaterClusters.h"
#include "UpdaterClustersImplicit.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
#include "IntegratorHPMCMonoImplicitNewGPU.h"
#include "ComputeFreeVolumeGPU.h"
#endif

namespace py = pybind11;

using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the base HPMCMono integrators
void export_union_sphinx(py::module& m)
    {
    export_IntegratorHPMCMono< ShapeUnion<ShapeSphinx> >(m, "IntegratorHPMCMonoSphinxUnion");
    export_IntegratorHPMCMonoImplicit< ShapeUnion<ShapeSphinx> >(m, "IntegratorHPMCMonoImplicitSphinxUnion");
    export_ComputeFreeVolume< ShapeUnion<ShapeSphinx> >(m, "ComputeFreeVolumeSphinxUnion");
    // export_AnalyzerSDF< ShapeUnion<ShapeSphinx> >(m, "AnalyzerSDFSphinxUnion");
    export_UpdaterMuVT< ShapeUnion<ShapeSphinx> >(m, "UpdaterMuVTSphinxUnion");
    export_UpdaterClusters<ShapeUnion<ShapeSphinx> >(m, "UpdaterClustersSphinxUnion");
    export_UpdaterClustersImplicit<ShapeUnion<ShapeSphinx>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeSphinx> > >(m, "UpdaterClustersImplicitSphinxUnion");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphinx>, IntegratorHPMCMonoImplicit<ShapeUnion<ShapeSphinx> > >(m, "UpdaterMuVTImplicitSphinxUnion");

    export_ExternalFieldInterface<ShapeUnion<ShapeSphinx> >(m, "ExternalFieldSphinxUnion");
    export_LatticeField<ShapeUnion<ShapeSphinx> >(m, "ExternalFieldLatticeSphinxUnion");
    export_ExternalFieldComposite<ShapeUnion<ShapeSphinx> >(m, "ExternalFieldCompositeSphinxUnion");
    export_RemoveDriftUpdater<ShapeUnion<ShapeSphinx> >(m, "RemoveDriftUpdaterSphinxUnion");
    export_ExternalFieldWall<ShapeUnion<ShapeSphinx> >(m, "WallSphinxUnion");
    export_UpdaterExternalFieldWall<ShapeUnion<ShapeSphinx> >(m, "UpdaterExternalFieldWallSphinxUnion");

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU< ShapeUnion<ShapeSphinx> >(m, "IntegratorHPMCMonoGPUSphinxUnion");
    export_IntegratorHPMCMonoImplicitGPU< ShapeUnion<ShapeSphinx> >(m, "IntegratorHPMCMonoImplicitGPUSphinxUnion");
    export_IntegratorHPMCMonoImplicitNewGPU< ShapeUnion<ShapeSphinx> >(m, "IntegratorHPMCMonoImplicitNewGPUSphinxUnion");
    export_ComputeFreeVolumeGPU< ShapeUnion<ShapeSphinx> >(m, "ComputeFreeVolumeGPUSphinxUnion");

    #endif
    #endif
    }

}
