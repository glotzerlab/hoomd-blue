// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SystemDataSnapshot.cc
 * \brief Defines the mpcd::SystemDataSnapshot class
 */

#include "SystemDataSnapshot.h"

namespace hoomd
    {
void mpcd::SystemDataSnapshot::replicate(unsigned int nx, unsigned int ny, unsigned int nz)
    {
    assert(nx > 0);
    assert(ny > 0);
    assert(nz > 0);

    // Update global box
    const BoxDim old_box = m_global_box;
    Scalar3 L = m_global_box.getL();
    L.x *= (Scalar)nx;
    L.y *= (Scalar)ny;
    L.z *= (Scalar)nz;
    m_global_box.setL(L);

    particles->replicate(nx, ny, nz, old_box, m_global_box);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SystemDataSnapshot(pybind11::module& m)
    {
    pybind11::class_<mpcd::SystemDataSnapshot, std::shared_ptr<mpcd::SystemDataSnapshot>>(
        m,
        "SystemDataSnapshot")
        .def(pybind11::init<std::shared_ptr<hoomd::SystemDefinition>>())
        .def("replicate", &mpcd::SystemDataSnapshot::replicate)
        .def_readonly("particles", &mpcd::SystemDataSnapshot::particles);
    }

    } // end namespace hoomd
