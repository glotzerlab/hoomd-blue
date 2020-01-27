// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file HOOMDMath.cc
    \brief Defines code needed for common math operations
 */

#include "HOOMDMath.h"
#include "VectorMath.h"

namespace py = pybind11;

void export_hoomd_math_functions(py::module& m)
    {
    // The use of shared_ptr's for exporting CUDA vector types is a workaround
    // see http://stackoverflow.com/questions/13177573/how-to-expose-aligned-class-with-boost-python

    // these clumsy property lambdas are necessary to make pybind11 interoperate with HIP vector types
    #ifdef SINGLE_PRECISION
    py::class_<double2, std::shared_ptr<double2> >(m,"double2")
        .def(py::init<>())
        .def_property("x", [](const double2& s) { return (double) s.x; }, [](double2 & s, double v) { s.x = v; })
        .def_property("y", [](const double2& s) { return (double) s.y; }, [](double2 & s, double v) { s.y = v; })
        ;
    py::class_<double3, std::shared_ptr<double3> >(m,"double3")
        .def(py::init<>())
        .def_property("x", [](const double3& s) { return (double) s.x; }, [](double3 & s, double v) { s.x = v; })
        .def_property("y", [](const double3& s) { return (double) s.y; }, [](double3 & s, double v) { s.y = v; })
        .def_property("z", [](const double3& s) { return (double) s.z; }, [](double3 & s, double v) { s.z = v; })
        ;
    py::class_<double4, std::shared_ptr<double4> >(m,"double4")
        .def(py::init<>())
        .def_property("x", [](const double4& s) { return (double) s.x; }, [](double4 & s, double v) { s.x = v; })
        .def_property("y", [](const double4& s) { return (double) s.y; }, [](double4 & s, double v) { s.y = v; })
        .def_property("z", [](const double4& s) { return (double) s.z; }, [](double4 & s, double v) { s.z = v; })
        .def_property("w", [](const double4& s) { return (double) s.w; }, [](double4 & s, double v) { s.w = v; })
        ;
    #else
    py::class_<float2, std::shared_ptr<float2> >(m,"float2")
        .def(py::init<>())
        .def_property("x", [](const float2& s) { return (float) s.x; }, [](float2 & s, float v) { s.x = v; })
        .def_property("y", [](const float2& s) { return (float) s.y; }, [](float2 & s, float v) { s.y = v; })
        ;
    py::class_<float3, std::shared_ptr<float3> >(m,"float3")
        .def(py::init<>())
        .def_property("x", [](const float3& s) { return (float) s.x; }, [](float3 & s, float v) { s.x = v; })
        .def_property("y", [](const float3& s) { return (float) s.y; }, [](float3 & s, float v) { s.y = v; })
        .def_property("z", [](const float3& s) { return (float) s.z; }, [](float3 & s, float v) { s.z = v; })
        ;
    py::class_<float4, std::shared_ptr<float4> >(m,"float4")
        .def(py::init<>())
        .def_property("x", [](const float4& s) { return (float) s.x; }, [](float4 & s, float v) { s.x = v; })
        .def_property("y", [](const float4& s) { return (float) s.y; }, [](float4 & s, float v) { s.y = v; })
        .def_property("z", [](const float4& s) { return (float) s.z; }, [](float4 & s, float v) { s.z = v; })
        .def_property("w", [](const float4& s) { return (float) s.w; }, [](float4 & s, float v) { s.w = v; })
        ;
    #endif

    py::class_<Scalar2, std::shared_ptr<Scalar2> >(m,"Scalar2")
        .def(py::init<>())
        .def_property("x", [](const Scalar2& s) { return (Scalar) s.x; }, [](Scalar2 & s, Scalar v) { s.x = v; })
        .def_property("y", [](const Scalar2& s) { return (Scalar) s.y; }, [](Scalar2 & s, Scalar v) { s.y = v; })
        ;
    py::class_<Scalar3, std::shared_ptr<Scalar3> >(m,"Scalar3")
        .def(py::init<>())
        .def_property("x", [](const Scalar3& s) { return (Scalar) s.x; }, [](Scalar3 & s, Scalar v) { s.x = v; })
        .def_property("y", [](const Scalar3& s) { return (Scalar) s.y; }, [](Scalar3 & s, Scalar v) { s.y = v; })
        .def_property("z", [](const Scalar3& s) { return (Scalar) s.z; }, [](Scalar3 & s, Scalar v) { s.z = v; })
        ;
    py::class_<Scalar4, std::shared_ptr<Scalar4> >(m,"Scalar4")
        .def(py::init<>())
        .def_property("x", [](const Scalar4& s) { return (Scalar) s.x; }, [](Scalar4 & s, Scalar v) { s.x = v; })
        .def_property("y", [](const Scalar4& s) { return (Scalar) s.y; }, [](Scalar4 & s, Scalar v) { s.y = v; })
        .def_property("z", [](const Scalar4& s) { return (Scalar) s.z; }, [](Scalar4 & s, Scalar v) { s.z = v; })
        .def_property("w", [](const Scalar4& s) { return (Scalar) s.w; }, [](Scalar4 & s, Scalar v) { s.w = v; })
        ;

    py::class_<uint2, std::shared_ptr<uint2> >(m,"uint2")
        .def(py::init<>())
        .def_property("x", [](const uint2& s) { return (unsigned int) s.x; }, [](uint2 & s, unsigned int v) { s.x = v; })
        .def_property("y", [](const uint2& s) { return (unsigned int) s.y; }, [](uint2 & s, unsigned int v) { s.y = v; })
        ;
    py::class_<uint3, std::shared_ptr<uint3> >(m,"uint3")
        .def(py::init<>())
        .def_property("x", [](const uint3& s) { return (unsigned int) s.x; }, [](uint3 & s, unsigned int v) { s.x = v; })
        .def_property("y", [](const uint3& s) { return (unsigned int) s.y; }, [](uint3 & s, unsigned int v) { s.y = v; })
        .def_property("z", [](const uint3& s) { return (unsigned int) s.z; }, [](uint3 & s, unsigned int v) { s.z = v; })
        ;
    py::class_<uint4, std::shared_ptr<uint4> >(m,"uint4")
        .def(py::init<>())
        .def_property("x", [](const uint4& s) { return (unsigned int) s.x; }, [](uint4 & s, unsigned int v) { s.x = v; })
        .def_property("y", [](const uint4& s) { return (unsigned int) s.y; }, [](uint4 & s, unsigned int v) { s.y = v; })
        .def_property("z", [](const uint4& s) { return (unsigned int) s.z; }, [](uint4 & s, unsigned int v) { s.z = v; })
        .def_property("w", [](const uint4& s) { return (unsigned int) s.w; }, [](uint4 & s, unsigned int v) { s.w = v; })
        ;

    py::class_<int2, std::shared_ptr<int2> >(m,"int2")
        .def(py::init<>())
        .def_property("x", [](const int2& s) { return (int) s.x; }, [](int2 & s, int v) { s.x = v; })
        .def_property("y", [](const int2& s) { return (int) s.y; }, [](int2 & s, int v) { s.y = v; })
        ;
    py::class_<int3, std::shared_ptr<int3> >(m,"int3")
        .def(py::init<>())
        .def_property("x", [](const int3& s) { return (int) s.x; }, [](int3 & s, int v) { s.x = v; })
        .def_property("y", [](const int3& s) { return (int) s.y; }, [](int3 & s, int v) { s.y = v; })
        .def_property("z", [](const int3& s) { return (int) s.z; }, [](int3 & s, int v) { s.z = v; })
        ;
    py::class_<int4, std::shared_ptr<int4> >(m,"int4")
        .def(py::init<>())
        .def_property("x", [](const int4& s) { return (int) s.x; }, [](int4 & s, int v) { s.x = v; })
        .def_property("y", [](const int4& s) { return (int) s.y; }, [](int4 & s, int v) { s.y = v; })
        .def_property("z", [](const int4& s) { return (int) s.z; }, [](int4 & s, int v) { s.z = v; })
        .def_property("w", [](const int4& s) { return (int) s.w; }, [](int4 & s, int v) { s.w = v; })
        ;

    py::class_<char3, std::shared_ptr<char3> >(m,"char3")
        .def(py::init<>())
        .def_property("x", [](const char3& s) { return (char) s.x; }, [](char3 & s, char v) { s.x = v; })
        .def_property("y", [](const char3& s) { return (char) s.y; }, [](char3 & s, char v) { s.y = v; })
        .def_property("z", [](const char3& s) { return (char) s.z; }, [](char3 & s, char v) { s.z = v; })
        ;
 
    m.def("make_scalar2", &make_scalar2);
    m.def("make_scalar3", &make_scalar3);
    m.def("make_scalar4", &make_scalar4);
    m.def("make_uint2", &make_uint2);
    m.def("make_uint3", &make_uint3);
    m.def("make_uint4", &make_uint4);
    m.def("make_int2", &make_int2);
    m.def("make_int3", &make_int3);
    m.def("make_int4", &make_int4);
    m.def("make_char3", &make_char3);
    m.def("int_as_scalar", &__int_as_scalar);

    // entries from VectorMath.h
    py::class_< vec3<float>, std::shared_ptr<vec3<float> > >(m,"vec3_float")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &vec3<float>::x)
        .def_readwrite("y", &vec3<float>::y)
        .def_readwrite("z", &vec3<float>::z)
        ;

    py::class_< vec3<double>, std::shared_ptr<vec3<double> > >(m,"vec3_double")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &vec3<double>::x)
        .def_readwrite("y", &vec3<double>::y)
        .def_readwrite("z", &vec3<double>::z)
        ;

    py::class_< quat<float>, std::shared_ptr<quat<float> > >(m,"quat_float")
        .def(py::init<float, const vec3<float>&>())
        .def_readwrite("s", &quat<float>::s)
        .def_readwrite("v", &quat<float>::v)
        .def_static("fromAxisAngle", &quat<float>::fromAxisAngle)
        ;

    }
