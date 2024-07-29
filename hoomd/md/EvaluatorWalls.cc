// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef __HIPCC__
#error This file cannot be compiled on the GPU.
#endif

#include "EvaluatorWalls.h"
#include "hoomd/ArrayView.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_wall_field(pybind11::module& m)
    {
    // Export the necessary ArrayView types to enable access in Python
    export_ArrayView<SphereWall>(m, "SphereArray");
    export_ArrayView<CylinderWall>(m, "CylinderArray");
    export_ArrayView<ConeWall>(m, "ConeArray");
    export_ArrayView<PlaneWall>(m, "PlaneArray");

    pybind11::class_<wall_type, std::shared_ptr<wall_type>>(m, "WallCollection")
        .def("_unsafe_create",
             []() -> std::shared_ptr<wall_type> { return std::make_shared<wall_type>(); })
        // The different get_*_list methods use ArrayView's (see hoomd/ArrayView.h for more info)
        // callback to ensure that the way_type object's sizes remain correct even during
        // modification.
        .def("get_sphere_list",
             [](wall_type& wall_list)
             {
                 return make_ArrayView(&wall_list.Spheres[0],
                                       MAX_N_SWALLS,
                                       wall_list.numSpheres,
                                       std::function<void(const ArrayView<SphereWall>*)>(
                                           [&wall_list](const ArrayView<SphereWall>* view) -> void {
                                               wall_list.numSpheres
                                                   = static_cast<unsigned int>(view->size);
                                           }));
             })
        .def("get_cylinder_list",
             [](wall_type& wall_list)
             {
                 return make_ArrayView(
                     &wall_list.Cylinders[0],
                     MAX_N_CWALLS,
                     wall_list.numCylinders,
                     std::function<void(const ArrayView<CylinderWall>*)>(
                         [&wall_list](const ArrayView<CylinderWall>* view) -> void
                         { wall_list.numCylinders = static_cast<unsigned int>(view->size); }));
             })
        .def("get_cone_list",
             [](wall_type& wall_list)
             {
                 return make_ArrayView(
                     &wall_list.Cones[0],
                     MAX_N_COWALLS,
                     wall_list.numCones,
                     std::function<void(const ArrayView<ConeWall>*)>(
                         [&wall_list](const ArrayView<ConeWall>* view) -> void
                         { wall_list.numCylinders = static_cast<unsigned int>(view->size); }));
             })
        .def("get_plane_list",
             [](wall_type& wall_list)
             {
                 return make_ArrayView(&wall_list.Planes[0],
                                       MAX_N_PWALLS,
                                       wall_list.numPlanes,
                                       std::function<void(const ArrayView<PlaneWall>*)>(
                                           [&wall_list](const ArrayView<PlaneWall>* view) -> void {
                                               wall_list.numPlanes
                                                   = static_cast<unsigned int>(view->size);
                                           }));
             })
        // These functions are not necessary for the Python interface but allow for more ready
        // testing of the ArrayView class and this exporting.
        .def("get_sphere", &wall_type::getSphere)
        .def("get_cylinder", &wall_type::getCylinder)
        .def("get_cone", &wall_type::getCone)
        .def("get_plane", &wall_type::getPlane)
        .def_property_readonly("num_spheres", &wall_type::getNumSpheres)
        .def_property_readonly("num_cylinders", &wall_type::getNumCylinders)
        .def_property_readonly("num_cones", &wall_type::getNumCones)
        .def_property_readonly("num_planes", &wall_type::getNumPlanes);
    }

    } // namespace detail
    } // namespace md
    } // namespace hoomd
