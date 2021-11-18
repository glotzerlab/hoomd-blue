// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: pschoenhoefer

#include "ManifoldGyroid.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
//! Exports the Gyroid manifold class to python
void export_ManifoldGyroid(pybind11::module& m)
    {
    pybind11::class_<ManifoldGyroid, std::shared_ptr<ManifoldGyroid>>(m, "ManifoldGyroid")
        .def(pybind11::init<int3, Scalar>())
        .def_property_readonly("N", &ManifoldGyroid::getN)
        .def_property_readonly("epsilon", &ManifoldGyroid::getEpsilon);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
