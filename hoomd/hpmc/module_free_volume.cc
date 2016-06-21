// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ComputeFreeVolume.h"
#ifdef ENABLE_CUDA
#include "ComputeFreeVolumeGPU.h"
#endif

#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSpheropolygon.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
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
void export_free_volume()
    {
    export_ComputeFreeVolume< ShapeSpheropolyhedron<8> >("ComputeFreeVolumeSpheropolyhedron8");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<16> >("ComputeFreeVolumeSpheropolyhedron16");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<32> >("ComputeFreeVolumeSpheropolyhedron32");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<64> >("ComputeFreeVolumeSpheropolyhedron64");
    export_ComputeFreeVolume< ShapeSpheropolyhedron<128> >("ComputeFreeVolumeSpheropolyhedron128");

    #ifdef ENABLE_CUDA
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<8> >("ComputeFreeVolumeGPUSpheropolyhedron8");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<16> >("ComputeFreeVolumeGPUSpheropolyhedron16");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<32> >("ComputeFreeVolumeGPUSpheropolyhedron32");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<64> >("ComputeFreeVolumeGPUSpheropolyhedron64");
    export_ComputeFreeVolumeGPU< ShapeSpheropolyhedron<128> >("ComputeFreeVolumeGPUSpheropolyhedron128");
    #endif
    }

}
