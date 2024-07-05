// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotential.h"

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::hpmc::PairPotential>>);

namespace hoomd::hpmc::detail
    {
void exportPairPotential(pybind11::module& m)
    {
    pybind11::class_<hpmc::PairPotential, std::shared_ptr<hpmc::PairPotential>>(m, "PairPotential")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParent", &hpmc::PairPotential::setParent);

    pybind11::bind_vector<std::vector<std::shared_ptr<PairPotential>>>(m, "PairPotentialList");
    }
    } // namespace hoomd::hpmc::detail
