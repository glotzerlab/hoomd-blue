// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldSphere.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Sphere manifold class to python
void export_ManifoldSphere(pybind11::module& m)
    {
    pybind11::class_<ManifoldSphere, std::shared_ptr<ManifoldSphere>>(m, "ManifoldSphere")
        .def(pybind11::init<Scalar, Scalar3>())
        .def_property_readonly("r", &ManifoldSphere::getR)
        .def_property_readonly("P", &ManifoldSphere::getP);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
