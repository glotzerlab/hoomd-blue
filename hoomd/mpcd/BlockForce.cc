// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BlockForce.cc
 * \brief Export MPCD BlockForce.
 */

#include "BlockForce.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {

void export_BlockForce(pybind11::module& m)
    {
    pybind11::class_<BlockForce, std::shared_ptr<BlockForce>>(m, "BlockForce")
        .def(pybind11::init<Scalar, Scalar, Scalar>())
        .def_property("force", &BlockForce::getForce, &BlockForce::setForce)
        .def_property("separation", &BlockForce::getSeparation, &BlockForce::setSeparation)
        .def_property("width", &BlockForce::getWidth, &BlockForce::setWidth);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
