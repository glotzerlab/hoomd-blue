// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

#include "ExternalPotential.h"

namespace hoomd::hpmc::detail
    {
void exportExternalPotential(pybind11::module& m)
    {
    pybind11::class_<ExternalPotential, std::shared_ptr<ExternalPotential>>(m, "ExternalPotential")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("totalEnergy",
             [](std::shared_ptr<ExternalPotential> self, uint64_t timestep)
             { return self->totalEnergy(timestep); });
    }
    } // namespace hoomd::hpmc::detail
