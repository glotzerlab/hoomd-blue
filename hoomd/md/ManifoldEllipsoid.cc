// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldEllipsoid.h"
#include <pybind11/pybind11.h>

//! Exports the Ellipsoid manifold class to python
void export_ManifoldEllipsoid(pybind11::module& m)
    {
    pybind11::class_< ManifoldEllipsoid, std::shared_ptr<ManifoldEllipsoid> >(m, "ManifoldEllipsoid")
    .def(pybind11::init<Scalar, Scalar, Scalar, Scalar3 >())
    .def("implicit_function", &ManifoldEllipsoid::implicit_function)
    .def("derivative", &ManifoldEllipsoid::derivative)
    ;
    }


