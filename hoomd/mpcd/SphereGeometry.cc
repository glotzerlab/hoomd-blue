// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SphereGeometry.cc
 * \brief Export function MPCD sphere geometry.
 */

#include "SphereGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_SphereGeometry(pybind11::module& m)
    {
    pybind11::class_<SphereGeometry, std::shared_ptr<SphereGeometry>>(
        m,
        SphereGeometry::getName().c_str())
        .def(pybind11::init<Scalar, bool>())
        .def_property_readonly("radius", &SphereGeometry::getRadius)
        .def_property_readonly("no_slip", &SphereGeometry::getNoSlip);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
