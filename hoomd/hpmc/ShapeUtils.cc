// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShapeUtils.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace hpmc
    {

template<class Shape> void export_MassPropertiesBase(pybind11::module& m, std::string name)
    {
    // export the base class.
    using detail::MassPropertiesBase;
    pybind11::class_<MassPropertiesBase<Shape>, std::shared_ptr<MassPropertiesBase<Shape>>>(
        m,
        name.c_str())
        .def(pybind11::init<>())
        .def("getVolume", &MassPropertiesBase<Shape>::getVolume)
        .def("getCenterOfMassElement", &MassPropertiesBase<Shape>::getCenterOfMassElement)
        .def("getInertiaTensor", &MassPropertiesBase<Shape>::getInertiaTensor)
        .def("getDetInertiaTensor", &MassPropertiesBase<Shape>::getDetInertiaTensor);
    }

template<class Shape> void export_MassProperties(pybind11::module& m, std::string name)
    {
    export_MassPropertiesBase<Shape>(m, name + "_base");
    // export the base class.
    using detail::MassProperties;
    using detail::MassPropertiesBase;
    pybind11::class_<MassProperties<Shape>,
                     std::shared_ptr<MassProperties<Shape>>,
                     MassPropertiesBase<Shape>>(m, name.c_str())
        .def(pybind11::init<const typename Shape::param_type&>());
    }

template void export_MassProperties<ShapeConvexPolyhedron>(pybind11::module& m, std::string name);
template void export_MassProperties<ShapeEllipsoid>(pybind11::module& m, std::string name);

    } // end namespace hpmc
    } // end namespace hoomd
