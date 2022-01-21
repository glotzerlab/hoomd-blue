// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExternalFieldWall.h"

PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::SphereWall>);
PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::CylinderWall>);
PYBIND11_MAKE_OPAQUE(std::vector<hoomd::hpmc::PlaneWall>);

void export_wall_classes(pybind11::module& m)
    {
    pybind11::class_<hoomd::hpmc::SphereWall>(m, "SphereWall")
        .def(pybind11::init(
                 [](hoomd::Scalar radius, pybind11::tuple origin)
                 {
                     return hoomd::hpmc::SphereWall(
                         radius,
                         hoomd::make_scalar3(origin[0].cast<hoomd::Scalar>(),
                                             origin[1].cast<hoomd::Scalar>(),
                                             origin[2].cast<hoomd::Scalar>()), );
                 }),
             pybind11::arg("radius"),
             pybind11::arg("origin"))
        .def_property_readonly("radius", [](const hoomd::hpmc::SphereWall& wall) { return wall.r; })
        .def_property_readonly(
            "origin",
            [](const hoomd::hpmc::SphereWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly("inside",
                               [](const hoomd::hpmc::SphereWall& wall) { return wall.inside; })
        .def_property_readonly("open",
                               [](const hoomd::hpmc::SphereWall& wall) { return wall.open; });

    pybind11::class_<hoomd::hpmc::CylinderWall>(m, "CylinderWall")
        .def(pybind11::init(
                 [](hoomd::Scalar radius, pybind11::tuple origin, pybind11::tuple z_orientation)
                 {
                     return hoomd::hpmc::CylinderWall(
                         radius,
                         hoomd::make_scalar3(origin[0].cast<hoomd::Scalar>(),
                                             origin[1].cast<hoomd::Scalar>(),
                                             origin[2].cast<hoomd::Scalar>()),
                         hoomd::make_scalar3(z_orientation[0].cast<hoomd::Scalar>(),
                                             z_orientation[1].cast<hoomd::Scalar>(),
                                             z_orientation[2].cast<hoomd::Scalar>()));
                 }),
             pybind11::arg("radius"),
             pybind11::arg("origin"),
             pybind11::arg("axis"),
             pybind11::arg("inside"),
             pybind11::arg("open"))
        .def_property_readonly("radius",
                               [](const hoomd::hpmc::CylinderWall& wall) { return wall.r; })
        .def_property_readonly(
            "origin",
            [](const hoomd::hpmc::CylinderWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly(
            "axis",
            [](const hoomd::hpmc::CylinderWall& wall)
            { return pybind11::make_tuple(wall.axis.x, wall.axis.y, wall.axis.z); });

    pybind11::class_<hoomd::hpmc::PlaneWall>(m, "PlaneWall")
        .def(pybind11::init(
                 [](pybind11::tuple origin, pybind11::tuple normal)
                 {
                     return hoomd::hpmc::PlaneWall(
                         hoomd::make_scalar3(origin[0].cast<hoomd::Scalar>(),
                                             origin[1].cast<hoomd::Scalar>(),
                                             origin[2].cast<hoomd::Scalar>()),
                         hoomd::make_scalar3(normal[0].cast<hoomd::Scalar>(),
                                             normal[1].cast<hoomd::Scalar>(),
                                             normal[2].cast<hoomd::Scalar>()));
                 }),
             pybind11::arg("origin"),
             pybind11::arg("normal"),
             pybind11::arg("open"))
        .def_property_readonly(
            "origin",
            [](const hoomd::hpmc::PlaneWall& wall)
            { return pybind11::make_tuple(wall.origin.x, wall.origin.y, wall.origin.z); })
        .def_property_readonly(
            "normal",
            [](const hoomd::hpmc::PlaneWall& wall)
            { return pybind11::make_tuple(wall.normal.x, wall.normal.y, wall.normal.z); })
        .def_property_readonly("open",
                               [](const hoomd::hpmc::PlaneWall& wall) { return wall.open; });
    }

void export_wall_list(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<hoomd::hpmc::SphereWall>>(m, "SphereWallList");
    pybind11::bind_vector<std::vector<hoomd::hpmc::CylinderWall>>(m, "CylinderWallList");
    pybind11::bind_vector<std::vector<hoomd::hpmc::PlaneWall>>(m, "PlaneWallList");
    }
