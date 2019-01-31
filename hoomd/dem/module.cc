// Copyright (c) 2009-2019 The Regents of the University of Michigan
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
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

void export_params(pybind11::module& m);

void export_NF_WCA_2D(pybind11::module& m);
void export_NF_WCA_3D(pybind11::module& m);
void export_NF_SWCA_3D(pybind11::module& m);
void export_NF_SWCA_2D(pybind11::module& m);

PYBIND11_MODULE(_dem, m)
    {
    export_params(m);
    export_NF_WCA_2D(m);
    export_NF_WCA_3D(m);
    export_NF_SWCA_2D(m);
    export_NF_SWCA_3D(m);
    }

// Export all of the parameter wrapper objects to the python interface
void export_params(pybind11::module& m)
    {
    pybind11::class_<NoFriction<Scalar> >(m, "NoFriction")
        .def(pybind11::init());

    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar> > WCA;
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > SWCA;

    pybind11::class_<WCA>(m, "WCAPotential")
        .def(pybind11::init<Scalar, NoFriction<Scalar> >());
    pybind11::class_<SWCA>(m, "SWCAPotential")
        .def(pybind11::init<Scalar, NoFriction<Scalar> >());
    }
