// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"

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
#include "AnalyzerSDF.h"

#ifdef ENABLE_CUDA
#include "IntegratorHPMCMonoGPU.h"
#include "IntegratorHPMCMonoImplicitGPU.h"
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

//! Export the GPU integrators
void export_hpmc_gpu()
    {
    #ifdef ENABLE_CUDA
    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<8> >("IntegratorHPMCMonoGPUSpheropolyhedron8");
    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoGPUSpheropolyhedron16");
    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoGPUSpheropolyhedron32");
    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoGPUSpheropolyhedron64");
    export_IntegratorHPMCMonoGPU< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoGPUSpheropolyhedron128");

    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<8> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron8");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<16> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron16");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<32> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron32");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<64> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron64");
    export_IntegratorHPMCMonoImplicitGPU< ShapeSpheropolyhedron<128> >("IntegratorHPMCMonoImplicitGPUSpheropolyhedron128");

    #endif
    }
}
