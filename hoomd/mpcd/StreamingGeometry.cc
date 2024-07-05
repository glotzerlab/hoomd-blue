// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/StreamingGeometry.cc
 * \brief Export functions for MPCD streaming geometries.
 */

#include "StreamingGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_boundary(pybind11::module& m)
    {
    pybind11::enum_<boundary>(m, "boundary")
        .value("no_slip", boundary::no_slip)
        .value("slip", boundary::slip);
    }

void export_BulkGeometry(pybind11::module& m)
    {
    pybind11::class_<BulkGeometry, std::shared_ptr<BulkGeometry>>(m, "BulkGeometry")
        .def(pybind11::init<>());
    }

void export_SlitGeometry(pybind11::module& m)
    {
    pybind11::class_<SlitGeometry, std::shared_ptr<SlitGeometry>>(m, "SlitGeometry")
        .def(pybind11::init<Scalar, Scalar, boundary>())
        .def("getH", &SlitGeometry::getH)
        .def("getVelocity", &SlitGeometry::getVelocity)
        .def("getBoundaryCondition", &SlitGeometry::getBoundaryCondition);
    }

void export_SlitPoreGeometry(pybind11::module& m)
    {
    pybind11::class_<SlitPoreGeometry, std::shared_ptr<SlitPoreGeometry>>(m, "SlitPoreGeometry")
        .def(pybind11::init<Scalar, Scalar, boundary>())
        .def("getH", &SlitPoreGeometry::getH)
        .def("getL", &SlitPoreGeometry::getL)
        .def("getBoundaryCondition", &SlitPoreGeometry::getBoundaryCondition);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
