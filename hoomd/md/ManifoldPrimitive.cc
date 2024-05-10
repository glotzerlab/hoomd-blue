// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldPrimitive.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Primitive manifold class to python
void export_ManifoldPrimitive(pybind11::module& m)
    {
    pybind11::class_<ManifoldPrimitive, std::shared_ptr<ManifoldPrimitive>>(m, "ManifoldPrimitive")
        .def(pybind11::init<int3, Scalar>())
        .def_property_readonly("N", &ManifoldPrimitive::getN)
        .def_property_readonly("epsilon", &ManifoldPrimitive::getEpsilon);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
