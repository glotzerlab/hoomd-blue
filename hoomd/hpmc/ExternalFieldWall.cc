// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExternalFieldWall.h"

namespace hoomd
    {
namespace hpmc
    {
namespace detail
    {
void export_wall_classes(pybind11::module& m)
    {
    pybind11::class_<hoomd::hpmc::SphereWall>(m, "SphereWall")
        .def(pybind11::init<hoomd::Scalar, pybind11::tuple, bool>())
        .def_property_readonly("radius", &hoomd::hpmc::SphereWall::getRadius)
        .def_property_readonly("origin", &hoomd::hpmc::SphereWall::getOrigin)
        .def_property_readonly("inside", &hoomd::hpmc::SphereWall::getInside)
        .def_property_readonly("open", &hoomd::hpmc::SphereWall::getOpen);

    pybind11::class_<hoomd::hpmc::CylinderWall>(m, "CylinderWall")
        .def(pybind11::init<hoomd::Scalar, pybind11::tuple, pybind11::tuple, bool>())
        .def_property_readonly("radius", &hoomd::hpmc::CylinderWall::getRadius)
        .def_property_readonly("origin", &hoomd::hpmc::CylinderWall::getOrigin)
        .def_property_readonly("axis", &hoomd::hpmc::CylinderWall::getAxis);

    pybind11::class_<hoomd::hpmc::PlaneWall>(m, "PlaneWall")
        .def(pybind11::init<pybind11::tuple, pybind11::tuple>())
        .def_property_readonly("origin", &hoomd::hpmc::PlaneWall::getOrigin)
        .def_property_readonly("normal", &hoomd::hpmc::PlaneWall::getNormal);
    }

void export_wall_list(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<hoomd::hpmc::SphereWall>>(m, "SphereWallList");
    pybind11::bind_vector<std::vector<hoomd::hpmc::CylinderWall>>(m, "CylinderWallList");
    pybind11::bind_vector<std::vector<hoomd::hpmc::PlaneWall>>(m, "PlaneWallList");
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
