// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldCuboid.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Cuboid manifold class to python
void export_ManifoldCuboid(pybind11::module& m)
    {
    pybind11::class_<ManifoldCuboid, std::shared_ptr<ManifoldCuboid>>(m, "ManifoldCuboid")
        .def(pybind11::init<Scalar3, Scalar3>())
        .def_property_readonly("a", &ManifoldCuboid::getA)
        .def_property_readonly("P", &ManifoldCuboid::getP);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
