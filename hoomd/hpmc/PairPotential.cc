// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

#include "PairPotential.h"

namespace hoomd::hpmc::detail
    {
void exportPairPotential(pybind11::module& m)
    {
    pybind11::class_<hpmc::PairPotential, std::shared_ptr<hpmc::PairPotential>>(m, "PairPotential")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParent", &hpmc::PairPotential::setParent);
    }
    } // namespace hoomd::hpmc::detail
