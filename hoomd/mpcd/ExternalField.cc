// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ExternalField.cc
 * \brief Export functions for MPCD external fields.
 */

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_ExternalFieldPolymorph(pybind11::module& m)
    {
    typedef hoomd::GPUPolymorph<mpcd::ExternalField> ExternalFieldPolymorph;

    pybind11::class_<ExternalFieldPolymorph, std::shared_ptr<ExternalFieldPolymorph>>(
        m,
        "ExternalField")
        .def(pybind11::init<std::shared_ptr<const hoomd::ExecutionConfiguration>>())
        // each field needs to get at least one (factory) method
        .def("BlockForce",
             (void(ExternalFieldPolymorph::*)(Scalar, Scalar, Scalar))
                 & ExternalFieldPolymorph::reset<mpcd::BlockForce>)
        .def("ConstantForce",
             (void(ExternalFieldPolymorph::*)(Scalar3))
                 & ExternalFieldPolymorph::reset<mpcd::ConstantForce>)
        .def("SineForce",
             (void(ExternalFieldPolymorph::*)(Scalar, Scalar))
                 & ExternalFieldPolymorph::reset<mpcd::SineForce>);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
