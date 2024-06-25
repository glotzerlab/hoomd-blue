// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
        .def_property(
            "force",
            [](const ConstantForce& force)
            {
                const auto F = force.getForce();
                return pybind11::make_tuple(F.x, F.y, F.z);
            },
            [](ConstantForce& force, const pybind11::tuple& F)
            {
                force.setForce(make_scalar3(pybind11::cast<Scalar>(F[0]),
                                            pybind11::cast<Scalar>(F[1]),
                                            pybind11::cast<Scalar>(F[2])));
            });
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
