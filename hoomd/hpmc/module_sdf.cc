// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "AnalyzerSDF.h"
#include "UpdaterBoxNPT.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#endif

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
void export_sdf()
    {
    export_AnalyzerSDF< ShapeSphere >("AnalyzerSDFSphere");
    export_AnalyzerSDF< ShapeConvexPolygon >("AnalyzerSDFConvexPolygon");
    export_AnalyzerSDF< ShapeSimplePolygon >("AnalyzerSDFSimplePolygon");
    export_AnalyzerSDF< ShapeConvexPolyhedron<8> >("AnalyzerSDFConvexPolyhedron8");
    export_AnalyzerSDF< ShapeConvexPolyhedron<16> >("AnalyzerSDFConvexPolyhedron16");
    export_AnalyzerSDF< ShapeConvexPolyhedron<32> >("AnalyzerSDFConvexPolyhedron32");
    export_AnalyzerSDF< ShapeConvexPolyhedron<64> >("AnalyzerSDFConvexPolyhedron64");
    export_AnalyzerSDF< ShapeConvexPolyhedron<128> >("AnalyzerSDFConvexPolyhedron128");
    export_AnalyzerSDF< ShapeSpheropolyhedron<8> >("AnalyzerSDFSpheropolyhedron8");
    export_AnalyzerSDF< ShapeSpheropolyhedron<16> >("AnalyzerSDFSpheropolyhedron16");
    export_AnalyzerSDF< ShapeSpheropolyhedron<32> >("AnalyzerSDFSpheropolyhedron32");
    export_AnalyzerSDF< ShapeSpheropolyhedron<64> >("AnalyzerSDFSpheropolyhedron64");
    export_AnalyzerSDF< ShapeSpheropolyhedron<128> >("AnalyzerSDFSpheropolyhedron128");
    export_AnalyzerSDF< ShapeEllipsoid >("AnalyzerSDFEllipsoid");
    export_AnalyzerSDF< ShapeSpheropolygon >("AnalyzerSDFSpheropolygon");
    export_AnalyzerSDF< ShapeFacetedSphere >("AnalyzerSDFFacetedSphere");
    }

}
