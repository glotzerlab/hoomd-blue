// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExternalPotential.h"

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::hpmc::ExternalPotential>>);

namespace hoomd::hpmc::detail
    {
void exportExternalPotential(pybind11::module& m)
    {
    pybind11::class_<ExternalPotential, std::shared_ptr<ExternalPotential>>(m, "ExternalPotential")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("totalEnergy", &ExternalPotential::totalEnergy);

    pybind11::bind_vector<std::vector<std::shared_ptr<ExternalPotential>>>(m,
                                                                           "ExternalPotentialList");
    }
    } // namespace hoomd::hpmc::detail
