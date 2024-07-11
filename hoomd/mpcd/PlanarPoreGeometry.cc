// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/PlanarPoreGeometry.cc
 * \brief Export functions for MPCD planar pore geometry.
 */

#include "PlanarPoreGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
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
