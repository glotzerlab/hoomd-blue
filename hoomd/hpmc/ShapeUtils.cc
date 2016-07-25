#include "hoomd/hoomd_config.h"
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "ShapeUtils.h"

namespace hpmc{

namespace detail{
const unsigned int ConvexHull::invalid_index = -1;
const Scalar ConvexHull::zero = SMALL;
}

template<class Shape>
inline void export_massPropertiesBase(std::string name)
    {
    // export the base class.
    using detail::mass_properties_base;
    boost::python::class_<mass_properties_base<Shape>, boost::shared_ptr< mass_properties_base<Shape> >, boost::noncopyable >
        (   name.c_str(),
            boost::python::init< >()
        )
    .def("volume", &mass_properties_base<Shape>::getVolume)
    .def("center_of_mass", &mass_properties_base<Shape>::getCenterOfMassElement)
    .def("moment_of_inertia", &mass_properties_base<Shape>::getInertiaTensor)
    ;
    }

template<class Shape>
inline void export_massProperties(std::string name)
    {
    export_massPropertiesBase<Shape>(name + "_base");
    // export the base class.
    using detail::mass_properties;
    using detail::mass_properties_base;
    boost::python::class_<mass_properties<Shape>, boost::shared_ptr< mass_properties<Shape> >, boost::python::bases< mass_properties_base<Shape> >, boost::noncopyable >
        (   name.c_str(),
            boost::python::init<const typename Shape::param_type&>()
        )
    .def("index", &mass_properties<Shape>::getFaceIndex)
    .def("num_faces", &mass_properties<Shape>::getNumFaces)
    ;
    }


void export_shape_utils()
    {
    export_massProperties< ShapeConvexPolyhedron<8> >("MassPropertiesConvexPolyhedron8");
    export_massProperties< ShapeConvexPolyhedron<16> >("MassPropertiesConvexPolyhedron16");
    export_massProperties< ShapeConvexPolyhedron<32> >("MassPropertiesConvexPolyhedron32");
    export_massProperties< ShapeConvexPolyhedron<64> >("MassPropertiesConvexPolyhedron64");
    export_massProperties< ShapeConvexPolyhedron<128> >("MassPropertiesConvexPolyhedron128");
    }


}// end namespace hpmc
