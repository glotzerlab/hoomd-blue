// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Autotuned.h"

namespace hoomd
    {

namespace detail
    {
void export_Autotuned(pybind11::module& m)
    {
    pybind11::class_<Autotuned, std::shared_ptr<Autotuned>>(m, "Autotuned").def(pybind11::init<>());
    }
    } // end namespace detail

    } // end namespace hoomd
