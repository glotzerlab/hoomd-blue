// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeUtils.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {

void export_MassPropertiesBase(pybind11::module& m)
    {
    // export the base class.
    pybind11::class_<detail::MassPropertiesBase, std::shared_ptr<detail::MassPropertiesBase>>(
        m,
        "MassPropertiesBase")
        .def(pybind11::init<>())
        .def("getVolume", &detail::MassPropertiesBase::getVolume)
        .def("getDetInertiaTensor", &detail::MassPropertiesBase::getDetInertiaTensor);
    }

template<class Shape> void export_MassProperties(pybind11::module& m, std::string name)
    {
    // export the base class.
    pybind11::class_<detail::MassProperties<Shape>,
                     std::shared_ptr<detail::MassProperties<Shape>>,
                     detail::MassPropertiesBase>(m, name.c_str())
        .def(pybind11::init<const typename Shape::param_type&>());
    }

template void export_MassProperties<ShapeConvexPolyhedron>(pybind11::module& m, std::string name);
template void export_MassProperties<ShapeSpheropolyhedron>(pybind11::module& m, std::string name);
template void export_MassProperties<ShapeEllipsoid>(pybind11::module& m, std::string name);

    } // end namespace hpmc
    } // end namespace hoomd
