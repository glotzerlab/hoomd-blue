
#include "ShapeUtils.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc{

namespace detail{

const unsigned int ConvexHull::invalid_index = -1;
const Scalar ConvexHull::epsilon = Scalar(1e-7);

}

template<class Shape>
void export_massPropertiesBase(pybind11::module& m, std::string name)
    {
    // export the base class.
    using detail::mass_properties_base;
    pybind11::class_<mass_properties_base<Shape>, std::shared_ptr< mass_properties_base<Shape> > >( m, name.c_str() )
    .def(pybind11::init< >())
    .def("volume", &mass_properties_base<Shape>::getVolume)
    .def("center_of_mass", &mass_properties_base<Shape>::getCenterOfMassElement)
    .def("moment_of_inertia", &mass_properties_base<Shape>::getInertiaTensor)
    .def("determinant", &mass_properties_base<Shape>::getDeterminant)
    ;
    }

template<class Shape>
void export_massProperties(pybind11::module& m, std::string name)
    {
    export_massPropertiesBase<Shape>(m, name + "_base");
    // export the base class.
    using detail::mass_properties;
    using detail::mass_properties_base;
    pybind11::class_<mass_properties<Shape>, std::shared_ptr< mass_properties<Shape> > >( m, name.c_str(), pybind11::base< mass_properties_base<Shape> >())
    .def(pybind11::init<const typename Shape::param_type&>())
    .def("index", &mass_properties<Shape>::getFaceIndex)
    .def("num_faces", &mass_properties<Shape>::getNumFaces)
    ;
    }

template void export_massProperties< ShapeConvexPolyhedron >(pybind11::module& m, std::string name);

}// end namespace hpmc
