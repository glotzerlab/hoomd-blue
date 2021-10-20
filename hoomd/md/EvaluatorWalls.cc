#ifdef __HIPCC__
#error This file cannot be compiled on the GPU.
#endif

#include "hoomd/ArrayView.h"
# include "EvaluatorWalls.h"

void export_wall_field(pybind11::module& m)
    {
    // Export the necessary array_view types to enable access in Python
    export_array_view<SphereWall>(m, "SphereArray");
    export_array_view<CylinderWall>(m, "CylinderArray");
    export_array_view<PlaneWall>(m, "PlaneArray");

    pybind11::class_<wall_type, std::shared_ptr<wall_type>>(m, "WallCollection")
        .def("_unsafe_create",
             []() -> std::shared_ptr<wall_type> { return std::make_shared<wall_type>(); })
        // The different get_*_list methods use array_view's (see hoomd/ArrayView.h for more info)
        // callback to ensure that the way_type object's sizes remain correct even during
        // modification.
        .def("get_sphere_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Spheres[0],
                     MAX_N_SWALLS,
                     wall_list.numSpheres,
                     std::function<void(const array_view<SphereWall>*)>(
                         [&wall_list](const array_view<SphereWall>* view) -> void
                         { wall_list.numSpheres = static_cast<unsigned int>(view->size); }));
             })
        .def("get_cylinder_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Cylinders[0],
                     MAX_N_CWALLS,
                     wall_list.numCylinders,
                     std::function<void(const array_view<CylinderWall>*)>(
                         [&wall_list](const array_view<CylinderWall>* view) -> void
                         { wall_list.numCylinders = static_cast<unsigned int>(view->size); }));
             })
        .def("get_plane_list",
             [](wall_type& wall_list)
             {
                 return make_array_view(
                     &wall_list.Planes[0],
                     MAX_N_PWALLS,
                     wall_list.numPlanes,
                     std::function<void(const array_view<PlaneWall>*)>(
                         [&wall_list](const array_view<PlaneWall>* view) -> void
                         { wall_list.numPlanes = static_cast<unsigned int>(view->size); }));
             })
        // These functions are not necessary for the Python interface but allow for more ready
        // testing of the array_view class and this exporting.
        .def("get_sphere", &wall_type::getSphere)
        .def("get_cylinder", &wall_type::getCylinder)
        .def("get_plane", &wall_type::getPlane)
        .def_property_readonly("num_spheres", &wall_type::getNumSpheres)
        .def_property_readonly("num_cylinders", &wall_type::getNumCylinders)
        .def_property_readonly("num_planes", &wall_type::getNumPlanes);
    }
