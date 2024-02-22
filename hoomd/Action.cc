// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Action.h"

namespace hoomd
    {

namespace detail
    {
void export_Action(pybind11::module& m)
    {
    pybind11::class_<Action, Autotuned, std::shared_ptr<Action>>(m, "Action")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
    } // end namespace detail

    } // end namespace hoomd
