// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldEllipsoid.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Ellipsoid manifold class to python
void export_ManifoldEllipsoid(pybind11::module& m)
    {
    pybind11::class_<ManifoldEllipsoid, std::shared_ptr<ManifoldEllipsoid>>(m, "ManifoldEllipsoid")
        .def(pybind11::init<Scalar, Scalar, Scalar, Scalar3>())
        .def_property_readonly("a", &ManifoldEllipsoid::getA)
        .def_property_readonly("b", &ManifoldEllipsoid::getB)
        .def_property_readonly("c", &ManifoldEllipsoid::getC)
        .def_property_readonly("P", &ManifoldEllipsoid::getP);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
