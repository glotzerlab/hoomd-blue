// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SineForce.cc
 * \brief Export MPCD SineForce.
 */

#include "SineForce.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {

void export_SineForce(pybind11::module& m)
    {
    pybind11::class_<SineForce, std::shared_ptr<SineForce>>(m, "SineForce")
        .def(pybind11::init<Scalar, Scalar>())
        .def_property("amplitude", &SineForce::getAmplitude, &SineForce::setAmplitude)
        .def_property("wavenumber", &SineForce::getWavenumber, &SineForce::setWavenumber);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
