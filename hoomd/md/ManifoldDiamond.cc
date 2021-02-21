// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldDiamond.h"
#include <pybind11/pybind11.h>

//! Exports the Diamond manifold class to python
void export_ManifoldDiamond(pybind11::module& m)
    {
    pybind11::class_< ManifoldDiamond, std::shared_ptr<ManifoldDiamond> >(m, "ManifoldDiamond")
    .def(pybind11::init<int, int, int, Scalar >())
    .def("implicit_function", &ManifoldDiamond::implicit_function)
    .def("derivative", &ManifoldDiamond::derivative)
    ;
    }

