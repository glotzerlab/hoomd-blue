// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CosineExpansionContractionGeometry.cc
 * \brief Export function MPCD cosine expansion contraction geometry.
 */

#include "CosineExpansionContractionGeometry.h"

namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_CosineExpansionContractionGeometry(pybind11::module& m)
    {
    pybind11::class_<CosineExpansionContractionGeometry,
                     std::shared_ptr<CosineExpansionContractionGeometry>>(
        m,
        CosineExpansionContractionGeometry::getName().c_str())
        .def(pybind11::init<Scalar, Scalar, Scalar, bool>())
        .def_property_readonly("expansion_separation",
                               &CosineExpansionContractionGeometry::getExpansionSeparation)
        .def_property_readonly("contraction_separation",
                               &CosineExpansionContractionGeometry::getContractionSeparation)
        .def_property_readonly("repeat_length",
                               &CosineExpansionContractionGeometry::getRepeatLength)
        .def_property_readonly("no_slip", &CosineExpansionContractionGeometry::getNoSlip);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
