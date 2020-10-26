// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingGeometry.cc
 * \brief Export functions for MPCD streaming geometries.
 */

#include "StreamingGeometry.h"

namespace mpcd
{
namespace detail
{

void export_boundary(pybind11::module& m)
    {
    namespace py = pybind11;
    py::enum_<boundary>(m, "boundary")
        .value("no_slip", boundary::no_slip)
        .value("slip", boundary::slip);
    }

void export_BulkGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<BulkGeometry, std::shared_ptr<BulkGeometry> >(m, "BulkGeometry")
        .def(py::init<>());
    }

void export_SlitGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SlitGeometry, std::shared_ptr<SlitGeometry> >(m, "SlitGeometry")
        .def(py::init<Scalar, Scalar, boundary>())
        .def("getH", &SlitGeometry::getH)
        .def("getVelocity", &SlitGeometry::getVelocity)
        .def("getBoundaryCondition", &SlitGeometry::getBoundaryCondition);
    }

void export_SlitPoreGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SlitPoreGeometry, std::shared_ptr<SlitPoreGeometry> >(m, "SlitPoreGeometry")
        .def(py::init<Scalar, Scalar, boundary>())
        .def("getH", &SlitPoreGeometry::getH)
        .def("getL", &SlitPoreGeometry::getL)
        .def("getBoundaryCondition", &SlitPoreGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace mpcd
