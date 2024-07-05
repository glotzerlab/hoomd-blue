// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PatchEnergyJIT.h"
#include "PatchEnergyJITUnion.h"

#include "ExternalFieldJIT.h"

#include "hoomd/hpmc/ShapeConvexPolygon.h"
#include "hoomd/hpmc/ShapeConvexPolyhedron.h"
#include "hoomd/hpmc/ShapeEllipsoid.h"
#include "hoomd/hpmc/ShapeFacetedEllipsoid.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/hpmc/ShapeSimplePolygon.h"
#include "hoomd/hpmc/ShapeSphere.h"
#include "hoomd/hpmc/ShapeSpheropolygon.h"
#include "hoomd/hpmc/ShapeSpheropolyhedron.h"
#include "hoomd/hpmc/ShapeSphinx.h"

#include <string>

#ifdef ENABLE_HIP
#include "PatchEnergyJITGPU.h"
#include "PatchEnergyJITUnionGPU.h"
#endif

#include <pybind11/pybind11.h>

using namespace hoomd::hpmc;
using namespace hoomd::hpmc::detail;

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
 create the hoomd python module and define the exports here.
 */

PYBIND11_MODULE(_jit, m)
    {
    export_PatchEnergyJIT(m);
    export_PatchEnergyJITUnion(m);

    export_ExternalFieldJIT<ShapeSphere>(m, "ExternalFieldJITSphere");
    export_ExternalFieldJIT<ShapeConvexPolygon>(m, "ExternalFieldJITConvexPolygon");
    export_ExternalFieldJIT<ShapePolyhedron>(m, "ExternalFieldJITPolyhedron");
    export_ExternalFieldJIT<ShapeConvexPolyhedron>(m, "ExternalFieldJITConvexPolyhedron");
    export_ExternalFieldJIT<ShapeSpheropolyhedron>(m, "ExternalFieldJITSpheropolyhedron");
    export_ExternalFieldJIT<ShapeSpheropolygon>(m, "ExternalFieldJITSpheropolygon");
    export_ExternalFieldJIT<ShapeSimplePolygon>(m, "ExternalFieldJITSimplePolygon");
    export_ExternalFieldJIT<ShapeEllipsoid>(m, "ExternalFieldJITEllipsoid");
    export_ExternalFieldJIT<ShapeFacetedEllipsoid>(m, "ExternalFieldJITFacetedEllipsoid");
    export_ExternalFieldJIT<ShapeSphinx>(m, "ExternalFieldJITSphinx");

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    export_PatchEnergyJITGPU(m);
    export_PatchEnergyJITUnionGPU(m);
#endif
    }
