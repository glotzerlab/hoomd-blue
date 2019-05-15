// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ExternalField.cc
 * \brief Export functions for MPCD external fields.
 */

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.h"

namespace mpcd
{
namespace detail
{

void export_ExternalFieldPolymorph(pybind11::module& m)
    {
    namespace py = pybind11;
    typedef hoomd::GPUPolymorph<mpcd::ExternalField> ExternalFieldPolymorph;

    py::class_<ExternalFieldPolymorph, std::shared_ptr<ExternalFieldPolymorph>>(m, "ExternalField")
        .def(py::init<std::shared_ptr<const ::ExecutionConfiguration>>())
        // each field needs to get at least one (factory) method
        .def("BlockForce", (void (ExternalFieldPolymorph::*)(Scalar,Scalar,Scalar)) &ExternalFieldPolymorph::reset<mpcd::BlockForce>)
        .def("ConstantForce", (void (ExternalFieldPolymorph::*)(Scalar3)) &ExternalFieldPolymorph::reset<mpcd::ConstantForce>)
        .def("SineForce", (void (ExternalFieldPolymorph::*)(Scalar,Scalar)) &ExternalFieldPolymorph::reset<mpcd::SineForce>);
    }

} // end namespace detail
} // end namespace mpcd
