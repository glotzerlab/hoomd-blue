// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: pschoenhoefer

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
