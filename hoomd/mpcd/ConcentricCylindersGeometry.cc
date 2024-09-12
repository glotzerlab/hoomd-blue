// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ConcentricCylindersGeometry.cc
 * \brief Export function MPCD concentric cylinders geometry.
 */

#include "ConcentricCylindersGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_ConcentricCylindersGeometry(pybind11::module& m)
    {
    pybind11::class_<ConcentricCylindersGeometry, std::shared_ptr<ConcentricCylindersGeometry>>(
        m,
        ConcentricCylindersGeometry::getName().c_str())
        .def(pybind11::init<Scalar, Scalar, Scalar, bool>())
        .def_property_readonly("inner_radius", &ConcentricCylindersGeometry::getR0)
        .def_property_readonly("outer_radius", &ConcentricCylindersGeometry::getR1)
        .def_property_readonly("angular_speed", &ConcentricCylindersGeometry::getAngularSpeed)
        .def_property_readonly("no_slip", &ConcentricCylindersGeometry::getNoSlip);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
