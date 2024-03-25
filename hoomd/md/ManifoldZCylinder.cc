// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldZCylinder.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Cylinder manifold class to python
void export_ManifoldZCylinder(pybind11::module& m)
    {
    pybind11::class_<ManifoldZCylinder, std::shared_ptr<ManifoldZCylinder>>(m, "ManifoldZCylinder")
        .def(pybind11::init<Scalar, Scalar3>())
        .def_property_readonly("r", &ManifoldZCylinder::getR)
        .def_property_readonly("P", &ManifoldZCylinder::getP);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
