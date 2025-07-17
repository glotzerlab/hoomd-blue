// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ComputeFreeVolume.h"
#include "IntegratorHPMCMono.h"

#include "ComputeSDF.h"
#include "ShapeSphinx.h"

#include "ExternalFieldWall.h"

#include "UpdaterGCA.h"
#include "UpdaterMuVT.h"

#ifdef ENABLE_HIP
#include "ComputeFreeVolumeGPU.h"
#include "IntegratorHPMCMonoGPU.h"
#include "UpdaterGCAGPU.h"
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
//! Export the base HPMCMono integrators
void export_sphinx(pybind11::module& m)
    {
    export_IntegratorHPMCMono<ShapeSphinx>(m, "IntegratorHPMCMonoSphinx");
    export_ComputeFreeVolume<ShapeSphinx>(m, "ComputeFreeVolumeSphinx");
    export_ComputeSDF<ShapeSphinx>(m, "ComputeSDFSphinx");
    export_UpdaterMuVT<ShapeSphinx>(m, "UpdaterMuVTSphinx");
    export_UpdaterGCA<ShapeSphinx>(m, "UpdaterGCASphinx");

    export_ExternalFieldWall<ShapeSphinx>(m, "WallSphinx");

#ifdef ENABLE_HIP
#ifdef ENABLE_SPHINX_GPU

    export_IntegratorHPMCMonoGPU<ShapeSphinx>(m, "IntegratorHPMCMonoSphinxGPU");
    export_ComputeFreeVolumeGPU<ShapeSphinx>(m, "ComputeFreeVolumeSphinxGPU");
    export_UpdaterGCAGPU<ShapeSphinx>(m, "UpdaterGCASphinxGPU");

#endif
#endif
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd
