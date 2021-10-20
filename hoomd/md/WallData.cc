#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <functional>
#include <memory>

#include "WallData.h"
#include "hoomd/ArrayView.h"
// Export all wall data types into Python. This is needed to allow for syncing Python and C++
// list/array data structures containing walls for WallPotential objects.
void export_wall_data(pybind11::module& m)
    {
    pybind11::class_<SphereWall>(m, "SphereWall")
        .def(pybind11::init(
                 [](Scalar radius, pybind11::tuple origin, bool inside)
                 {
                     return SphereWall(radius,
                                       make_scalar3(origin[0].cast<Scalar>(),
                                                    origin[1].cast<Scalar>(),
                                                    origin[2].cast<Scalar>()),
                                       inside);
                 }),
             pybind11::arg("radius"),
             pybind11::arg("origin"),
             pybind11::arg("inside"))
        .def_property_readonly("radius", [](const SphereWall& wall) { return wall.r; })
        .def_property_readonly(
            "origin",
            [](const SphereWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly("inside", [](const SphereWall& wall) { return wall.inside; });

    pybind11::class_<CylinderWall>(m, "CylinderWall")
        .def(pybind11::init(
                 [](Scalar radius,
                    pybind11::tuple origin,
                    pybind11::tuple z_orientation,
                    bool inside)
                 {
                     return CylinderWall(radius,
                                         make_scalar3(origin[0].cast<Scalar>(),
                                                      origin[1].cast<Scalar>(),
                                                      origin[2].cast<Scalar>()),
                                         make_scalar3(z_orientation[0].cast<Scalar>(),
                                                      z_orientation[1].cast<Scalar>(),
                                                      z_orientation[2].cast<Scalar>()),
                                         inside);
                 }),
             pybind11::arg("radius"),
             pybind11::arg("origin"),
             pybind11::arg("axis"),
             pybind11::arg("inside"))
        .def_property_readonly("radius", [](const CylinderWall& wall) { return wall.r; })
        .def_property_readonly(
            "origin",
            [](const CylinderWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly(
            "axis",
            [](const CylinderWall& wall)
            { return pybind11::make_tuple(wall.axis.x, wall.axis.y, wall.axis.z); })
        .def_property_readonly("inside", [](const CylinderWall& wall) { return wall.inside; });

    pybind11::class_<PlaneWall>(m, "PlaneWall")
        .def(pybind11::init(
                 [](pybind11::tuple origin, pybind11::tuple normal, bool inside)
                 {
                     return PlaneWall(make_scalar3(origin[0].cast<Scalar>(),
                                                   origin[1].cast<Scalar>(),
                                                   origin[2].cast<Scalar>()),
                                      make_scalar3(normal[0].cast<Scalar>(),
                                                   normal[1].cast<Scalar>(),
                                                   normal[2].cast<Scalar>()),
                                      inside);
                 }),
             pybind11::arg("origin"),
             pybind11::arg("normal"),
             pybind11::arg("inside"))
        .def_property_readonly(
            "origin",
            [](const PlaneWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly(
            "normal",
            [](const PlaneWall& wall)
            { return pybind11::make_tuple(wall.normal.x, wall.normal.y, wall.normal.z); })
        .def_property_readonly("inside", [](const PlaneWall& wall) { return wall.inside; });
    }
