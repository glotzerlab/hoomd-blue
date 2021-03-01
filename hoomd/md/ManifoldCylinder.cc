// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldCylinder.h"
#include <pybind11/pybind11.h>


//! Exports the Cylinder manifold class to python
void export_ManifoldCylinder(pybind11::module& m)
    {
    pybind11::class_< ManifoldCylinder, std::shared_ptr<ManifoldCylinder> >(m, "ManifoldCylinder")
    .def(pybind11::init<Scalar, Scalar3 >())
    ;
    }
