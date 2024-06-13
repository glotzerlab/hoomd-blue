// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/NoForce.cc
 * \brief Export MPCD NoForce.
 */

#include "NoForce.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_NoForce(pybind11::module& m)
    {
    pybind11::class_<NoForce, std::shared_ptr<NoForce>>(m, "NoForce").def(pybind11::init<>());
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
