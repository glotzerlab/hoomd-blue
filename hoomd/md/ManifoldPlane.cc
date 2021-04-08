// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldPlane.h"
#include <pybind11/pybind11.h>



//! Exports the Plane manifold class to python
void export_ManifoldPlane(pybind11::module& m)
    {
    pybind11::class_< ManifoldPlane, std::shared_ptr<ManifoldPlane> >(m, "ManifoldPlane")
    .def(pybind11::init<Scalar >())
    .def_property_readonly("shift", &ManifoldPlane::getShift)
    ;
    }
