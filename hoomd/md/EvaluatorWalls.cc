#ifdef __HIPCC__
#error This file cannot be compiled on the GPU.
#endif

#include "EvaluatorWalls.h"
#include "hoomd/ArrayView.h"

void hoomd::md::export_wall_field(pybind11::module& m)
    {
    // Export the necessary ArrayView types to enable access in Python
    export_ArrayView<hoomd::md::SphereWall>(m, "SphereArray");
    export_ArrayView<hoomd::md::CylinderWall>(m, "CylinderArray");
    export_ArrayView<hoomd::md::PlaneWall>(m, "PlaneArray");

    pybind11::class_<hoomd::md::wall_type, std::shared_ptr<hoomd::md::wall_type>>(m,
                                                                                  "WallCollection")
        .def("_unsafe_create",
             []() -> std::shared_ptr<hoomd::md::wall_type>
             { return std::make_shared<hoomd::md::wall_type>(); })
        // The different get_*_list methods use ArrayView's (see hoomd/ArrayView.h for more info)
        // callback to ensure that the way_type object's sizes remain correct even during
        // modification.
        .def("get_sphere_list",
             [](hoomd::md::wall_type& wall_list)
             {
                 return make_ArrayView(
                     &wall_list.Spheres[0],
                     MAX_N_SWALLS,
                     wall_list.numSpheres,
                     std::function<void(const ArrayView<hoomd::md::SphereWall>*)>(
                         [&wall_list](const ArrayView<hoomd::md::SphereWall>* view) -> void
                         { wall_list.numSpheres = static_cast<unsigned int>(view->size); }));
             })
        .def("get_cylinder_list",
             [](hoomd::md::wall_type& wall_list)
             {
                 return make_ArrayView(
                     &wall_list.Cylinders[0],
                     MAX_N_CWALLS,
                     wall_list.numCylinders,
                     std::function<void(const ArrayView<hoomd::md::CylinderWall>*)>(
                         [&wall_list](const ArrayView<hoomd::md::CylinderWall>* view) -> void
                         { wall_list.numCylinders = static_cast<unsigned int>(view->size); }));
             })
        .def("get_plane_list",
             [](hoomd::md::wall_type& wall_list)
             {
                 return make_ArrayView(
                     &wall_list.Planes[0],
                     MAX_N_PWALLS,
                     wall_list.numPlanes,
                     std::function<void(const ArrayView<hoomd::md::PlaneWall>*)>(
                         [&wall_list](const ArrayView<hoomd::md::PlaneWall>* view) -> void
                         { wall_list.numPlanes = static_cast<unsigned int>(view->size); }));
             })
        // These functions are not necessary for the Python interface but allow for more ready
        // testing of the ArrayView class and this exporting.
        .def("get_sphere", &hoomd::md::wall_type::getSphere)
        .def("get_cylinder", &hoomd::md::wall_type::getCylinder)
        .def("get_plane", &hoomd::md::wall_type::getPlane)
        .def_property_readonly("num_spheres", &hoomd::md::wall_type::getNumSpheres)
        .def_property_readonly("num_cylinders", &hoomd::md::wall_type::getNumCylinders)
        .def_property_readonly("num_planes", &hoomd::md::wall_type::getNumPlanes);
    }
