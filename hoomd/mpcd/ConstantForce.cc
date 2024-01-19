// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ConstantForce.cc
 * \brief Export MPCD ConstantForce.
 */

#include "ConstantForce.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {

void export_ConstantForce(pybind11::module& m)
    {
    pybind11::class_<ConstantForce, std::shared_ptr<ConstantForce>>(m, "ConstantForce")
        .def(pybind11::init<Scalar3>())
        .def_property("force", &ConstantForce::getForce, &ConstantForce::setForce);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
