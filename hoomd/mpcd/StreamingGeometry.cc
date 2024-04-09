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

void export_ParallelPlateGeometry(pybind11::module& m)
    {
    pybind11::class_<ParallelPlateGeometry, std::shared_ptr<ParallelPlateGeometry>>(
        m,
        ParallelPlateGeometry::getName().c_str())
        .def(pybind11::init<Scalar, Scalar, bool>())
        .def_property_readonly("separation", &ParallelPlateGeometry::getSeparation)
        .def_property_readonly("speed", &ParallelPlateGeometry::getSpeed)
        .def_property_readonly("no_slip", &ParallelPlateGeometry::getNoSlip);
    }

void export_PlanarPoreGeometry(pybind11::module& m)
    {
    pybind11::class_<PlanarPoreGeometry, std::shared_ptr<PlanarPoreGeometry>>(
        m,
        PlanarPoreGeometry::getName().c_str())
        .def(pybind11::init<Scalar, Scalar, bool>())
        .def_property_readonly("separation", &PlanarPoreGeometry::getSeparation)
        .def_property_readonly("length", &PlanarPoreGeometry::getLength)
        .def_property_readonly("no_slip", &PlanarPoreGeometry::getNoSlip);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
