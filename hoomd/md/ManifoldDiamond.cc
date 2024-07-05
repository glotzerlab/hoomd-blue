// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ManifoldDiamond.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Diamond manifold class to python
void export_ManifoldDiamond(pybind11::module& m)
    {
    pybind11::class_<ManifoldDiamond, std::shared_ptr<ManifoldDiamond>>(m, "ManifoldDiamond")
        .def(pybind11::init<int3, Scalar>())
        .def_property_readonly("N", &ManifoldDiamond::getN)
        .def_property_readonly("epsilon", &ManifoldDiamond::getEpsilon);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
