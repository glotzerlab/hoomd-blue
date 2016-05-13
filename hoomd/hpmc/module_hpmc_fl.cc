// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "IntegratorHPMCMono_FL.h"

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeUnion.h"
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

//! Export the Frenkel-Ladd integrators
void export_hpmc_fl()
    {
    export_IntegratorHPMCMono_FL< ShapeSphere >("IntegratorHPMCMono_FLSphere");
    export_IntegratorHPMCMono_FL< ShapeConvexPolygon >("IntegratorHPMCMono_FLConvexPolygon");
    export_IntegratorHPMCMono_FL< ShapeSimplePolygon >("IntegratorHPMCMono_FLSimplePolygon");
    export_IntegratorHPMCMono_FL< ShapeConvexPolyhedron<8> >("IntegratorHPMCMono_FLConvexPolyhedron8");
    export_IntegratorHPMCMono_FL< ShapeConvexPolyhedron<16> >("IntegratorHPMCMono_FLConvexPolyhedron16");
    export_IntegratorHPMCMono_FL< ShapeConvexPolyhedron<32> >("IntegratorHPMCMono_FLConvexPolyhedron32");
    export_IntegratorHPMCMono_FL< ShapeConvexPolyhedron<64> >("IntegratorHPMCMono_FLConvexPolyhedron64");
    export_IntegratorHPMCMono_FL< ShapeConvexPolyhedron<128> >("IntegratorHPMCMono_FLConvexPolyhedron128");
    export_IntegratorHPMCMono_FL< ShapeSpheropolyhedron<8> >("IntegratorHPMCMono_FLSpheropolyhedron8");
    export_IntegratorHPMCMono_FL< ShapeSpheropolyhedron<16> >("IntegratorHPMCMono_FLSpheropolyhedron16");
    export_IntegratorHPMCMono_FL< ShapeSpheropolyhedron<32> >("IntegratorHPMCMono_FLSpheropolyhedron32");
    export_IntegratorHPMCMono_FL< ShapeSpheropolyhedron<64> >("IntegratorHPMCMono_FLSpheropolyhedron64");
    export_IntegratorHPMCMono_FL< ShapeSpheropolyhedron<128> >("IntegratorHPMCMono_FLSpheropolyhedron128");
    export_IntegratorHPMCMono_FL< ShapeEllipsoid >("IntegratorHPMCMono_FLEllipsoid");
    export_IntegratorHPMCMono_FL< ShapeSpheropolygon >("IntegratorHPMCMono_FLSpheropolygon");
    export_IntegratorHPMCMono_FL< ShapeFacetedSphere >("IntegratorHPMCMono_FLFacetedSphere");
    export_IntegratorHPMCMono_FL< ShapeUnion<ShapeSphere> >("IntegratorHPMCMono_FLSphereUnion");
    }

}
