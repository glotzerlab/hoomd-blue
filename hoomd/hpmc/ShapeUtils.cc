// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
#include "ShapeUtils.h"
#include <pybind11/pybind11.h>

namespace hpmc{

namespace detail{


}

template<class Shape>
void export_MassPropertiesBase(pybind11::module& m, std::string name)
    {
    // export the base class.
    using detail::MassPropertiesBase;
    pybind11::class_<MassPropertiesBase<Shape>, std::shared_ptr< MassPropertiesBase<Shape> > >( m, name.c_str() )
    .def(pybind11::init< >())
    .def("volume", &MassPropertiesBase<Shape>::getVolume)
    .def("center_of_mass", &MassPropertiesBase<Shape>::getCenterOfMassElement)
    .def("moment_of_inertia", &MassPropertiesBase<Shape>::getInertiaTensor)
    .def("determinant", &MassPropertiesBase<Shape>::getDeterminant)
    ;
    }

template<class Shape>
void export_MassProperties(pybind11::module& m, std::string name)
    {
    export_MassPropertiesBase<Shape>(m, name + "_base");
    // export the base class.
    using detail::MassProperties;
    using detail::MassPropertiesBase;
    pybind11::class_<MassProperties<Shape>, std::shared_ptr< MassProperties<Shape> >, MassPropertiesBase<Shape> >
    ( m, name.c_str())
    .def(pybind11::init<const typename Shape::param_type&>())
    .def("vertices", &MassProperties<Shape>::getFaceVertices)
    .def("num_faces", &MassProperties<Shape>::getNumFaces)
    ;
    }

template void export_MassProperties< ShapeConvexPolyhedron >(pybind11::module& m, std::string name);

}// end namespace hpmc
