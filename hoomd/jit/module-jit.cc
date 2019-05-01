// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PatchEnergyJIT.h"
#include "PatchEnergyJITUnion.h"

//#include "hoomd/hpmc/IntegratorHPMC.h"
//#include "hoomd/hpmc/IntegratorHPMCMono.h"
//#include "hoomd/hpmc/IntegratorHPMCMonoImplicit.h"
#include "ExternalFieldJIT.h"
//#include "ExternalFieldJIT.cc"

#include "hoomd/hpmc/ShapeSphere.h"
#include "hoomd/hpmc/ShapeConvexPolygon.h"
#include "hoomd/hpmc/ShapePolyhedron.h"
#include "hoomd/hpmc/ShapeConvexPolyhedron.h"
#include "hoomd/hpmc/ShapeSpheropolyhedron.h"
#include "hoomd/hpmc/ShapeSpheropolygon.h"
#include "hoomd/hpmc/ShapeSimplePolygon.h"
#include "hoomd/hpmc/ShapeEllipsoid.h"
#include "hoomd/hpmc/ShapeFacetedEllipsoid.h"
#include "hoomd/hpmc/ShapeSphinx.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hpmc;
using namespace hpmc::detail;

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
    }
