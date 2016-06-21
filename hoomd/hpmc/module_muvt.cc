// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "UpdaterMuVT.h"
#include "UpdaterMuVTImplicit.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"

/*! \file module.cc
    \brief Export classes to python
*/

// Include boost.python to do the exporting
#include <boost/python.hpp>

using namespace boost::python;
using namespace hpmc;

using namespace hpmc::detail;

namespace hpmc
{

//! Export the SDF analyzers
void export_muvt()
    {
    export_UpdaterMuVT< ShapeConvexPolyhedron<8> >("UpdaterMuVTConvexPolyhedron8");
    export_UpdaterMuVT< ShapeConvexPolyhedron<16> >("UpdaterMuVTConvexPolyhedron16");
    export_UpdaterMuVT< ShapeConvexPolyhedron<32> >("UpdaterMuVTConvexPolyhedron32");
    export_UpdaterMuVT< ShapeConvexPolyhedron<64> >("UpdaterMuVTConvexPolyhedron64");
    export_UpdaterMuVT< ShapeConvexPolyhedron<128> >("UpdaterMuVTConvexPolyhedron128");
    export_UpdaterMuVT< ShapeSpheropolyhedron<8> >("UpdaterMuVTSpheropolyhedron8");
    export_UpdaterMuVT< ShapeSpheropolyhedron<16> >("UpdaterMuVTSpheropolyhedron16");
    export_UpdaterMuVT< ShapeSpheropolyhedron<32> >("UpdaterMuVTSpheropolyhedron32");
    export_UpdaterMuVT< ShapeSpheropolyhedron<64> >("UpdaterMuVTSpheropolyhedron64");
    export_UpdaterMuVT< ShapeSpheropolyhedron<128> >("UpdaterMuVTSpheropolyhedron128");
    export_UpdaterMuVT< ShapeUnion<ShapeSphere> >("UpdaterMuVTSphereUnion");

    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<8> >("UpdaterMuVTImplicitConvexPolyhedron8");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<16> >("UpdaterMuVTImplicitConvexPolyhedron16");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<32> >("UpdaterMuVTImplicitConvexPolyhedron32");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<64> >("UpdaterMuVTImplicitConvexPolyhedron64");
    export_UpdaterMuVTImplicit< ShapeConvexPolyhedron<128> >("UpdaterMuVTImplicitConvexPolyhedron128");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<8> >("UpdaterMuVTImplicitSpheropolyhedron8");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<16> >("UpdaterMuVTImplicitSpheropolyhedron16");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<32> >("UpdaterMuVTImplicitSpheropolyhedron32");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<64> >("UpdaterMuVTImplicitSpheropolyhedron64");
    export_UpdaterMuVTImplicit< ShapeSpheropolyhedron<128> >("UpdaterMuVTImplicitSpheropolyhedron128");
    export_UpdaterMuVTImplicit< ShapeUnion<ShapeSphere> >("UpdaterMuVTImplicitSphereUnion");
    }

}
