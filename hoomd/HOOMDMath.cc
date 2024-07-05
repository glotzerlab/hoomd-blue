// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file HOOMDMath.cc
    \brief Defines code needed for common math operations
 */

#include "HOOMDMath.h"
#include "VectorMath.h"

namespace hoomd
    {
namespace detail
    {
void export_hoomd_math_functions(pybind11::module& m)
    {
    // The use of shared_ptr's for exporting CUDA vector types is a workaround
    // see http://stackoverflow.com/questions/13177573/how-to-expose-aligned-class-with-boost-python

    // these clumsy property lambdas are necessary to make pybind11 interoperate with HIP vector
    // types
    pybind11::class_<double3, std::shared_ptr<double3>>(m, "double3")
        .def(pybind11::init<>())
        .def_property(
            "x",
            [](const double3& s) { return (double)s.x; },
            [](double3& s, double v) { s.x = v; })
        .def_property(
            "y",
            [](const double3& s) { return (double)s.y; },
            [](double3& s, double v) { s.y = v; })
        .def_property(
            "z",
            [](const double3& s) { return (double)s.z; },
            [](double3& s, double v) { s.z = v; });

    pybind11::class_<float3, std::shared_ptr<float3>>(m, "float3")
        .def(pybind11::init<>())
        .def_property(
            "x",
            [](const float3& s) { return (float)s.x; },
            [](float3& s, float v) { s.x = v; })
        .def_property(
            "y",
            [](const float3& s) { return (float)s.y; },
            [](float3& s, float v) { s.y = v; })
        .def_property(
            "z",
            [](const float3& s) { return (float)s.z; },
            [](float3& s, float v) { s.z = v; });

    pybind11::class_<int3, std::shared_ptr<int3>>(m, "int3")
        .def(pybind11::init<>())
        .def_property(
            "x",
            [](const int3& s) { return (int)s.x; },
            [](int3& s, int v) { s.x = v; })
        .def_property(
            "y",
            [](const int3& s) { return (int)s.y; },
            [](int3& s, int v) { s.y = v; })
        .def_property("z", [](const int3& s) { return (int)s.z; }, [](int3& s, int v) { s.z = v; });

    pybind11::class_<uint3, std::shared_ptr<uint3>>(m, "uint3")
        .def(pybind11::init<>())
        .def_property(
            "x",
            [](const uint3& s) { return (int)s.x; },
            [](uint3& s, int v) { s.x = v; })
        .def_property(
            "y",
            [](const uint3& s) { return (int)s.y; },
            [](uint3& s, int v) { s.y = v; })
        .def_property(
            "z",
            [](const uint3& s) { return (int)s.z; },
            [](uint3& s, int v) { s.z = v; });

    pybind11::class_<char3, std::shared_ptr<char3>>(m, "char3")
        .def(pybind11::init<>())
        .def_property(
            "x",
            [](const char3& s) { return (char)s.x; },
            [](char3& s, char v) { s.x = v; })
        .def_property(
            "y",
            [](const char3& s) { return (char)s.y; },
            [](char3& s, char v) { s.y = v; })
        .def_property(
            "z",
            [](const char3& s) { return (char)s.z; },
            [](char3& s, char v) { s.z = v; });

    m.def("make_scalar3", &make_scalar3);
    m.def("make_int3", &make_int3);
    m.def("make_char3", &make_char3);

    // entries from VectorMath.h
    pybind11::class_<vec3<float>, std::shared_ptr<vec3<float>>>(m, "vec3_float")
        .def(pybind11::init<float, float, float>())
        .def_readwrite("x", &vec3<float>::x)
        .def_readwrite("y", &vec3<float>::y)
        .def_readwrite("z", &vec3<float>::z);

    pybind11::class_<vec3<double>, std::shared_ptr<vec3<double>>>(m, "vec3_double")
        .def(pybind11::init<double, double, double>())
        .def_readwrite("x", &vec3<double>::x)
        .def_readwrite("y", &vec3<double>::y)
        .def_readwrite("z", &vec3<double>::z);
    }

    } // end namespace detail

    } // end namespace hoomd
