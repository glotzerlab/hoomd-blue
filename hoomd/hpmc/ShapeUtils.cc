
#include "ShapeUtils.h"
// #include <hoomd/extern/pybind/include/pybind11/pybind11.h>
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
    .def("index", &MassProperties<Shape>::getFaceIndex)
    .def("num_faces", &MassProperties<Shape>::getNumFaces)
    ;
    }

template void export_MassProperties< ShapeConvexPolyhedron >(pybind11::module& m, std::string name);

}// end namespace hpmc
