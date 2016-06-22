// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

// Include the defined classes that are to be exported to python
#include <hoomd/HOOMDMath.h>

#include "NoFriction.h"

#include "DEMEvaluator.h"
#include "WCAPotential.h"
#include "SWCAPotential.h"
#include "DEM2DForceCompute.h"
#include "DEM2DForceComputeGPU.h"
#include "DEM3DForceCompute.h"
#include "DEM3DForceComputeGPU.h"

#include <iterator>

// Include boost.python to do the exporting
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
using namespace boost::python;

void export_params();

void export_NF_WCA_2D();
void export_NF_WCA_3D();
void export_NF_SWCA_3D();
void export_NF_SWCA_2D();

BOOST_PYTHON_MODULE(_dem)
    {
    export_params();
    export_NF_WCA_2D();
    export_NF_WCA_3D();
    export_NF_SWCA_2D();
    export_NF_SWCA_3D();
    }

// Export all of the parameter wrapper objects to the python interface
void export_params()
    {
    class_<NoFriction<Scalar> >("NoFriction");

    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar> > WCA;
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > SWCA;

    class_<WCA>("WCAPotential", init<Scalar, NoFriction<Scalar> >());
    class_<SWCA>("SWCAPotential", init<Scalar, NoFriction<Scalar> >());
    }
