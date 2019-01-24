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
    #ifdef SINGLE_PRECISION
    py::class_<double2, std::shared_ptr<double2> >(m,"double2")
        .def(py::init<>())
        .def_readwrite("x", &double2::x)
        .def_readwrite("y", &double2::y)
        ;
    py::class_<double3, std::shared_ptr<double3> >(m,"double3")
        .def(py::init<>())
        .def_readwrite("x", &double3::x)
        .def_readwrite("y", &double3::y)
        .def_readwrite("z", &double3::z)
        ;
    py::class_<double4, std::shared_ptr<double4> >(m,"double4")
        .def(py::init<>())
        .def_readwrite("x", &double4::x)
        .def_readwrite("y", &double4::y)
        .def_readwrite("z", &double4::z)
        .def_readwrite("w", &double4::w)
        ;
    #else
    py::class_<float2, std::shared_ptr<float2> >(m,"float2")
        .def(py::init<>())
        .def_readwrite("x", &float2::x)
        .def_readwrite("y", &float2::y)
        ;
    py::class_<float3, std::shared_ptr<float3> >(m,"float3")
        .def(py::init<>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        ;
    py::class_<float4, std::shared_ptr<float4> >(m,"float4")
        .def(py::init<>())
        .def_readwrite("x", &float4::x)
        .def_readwrite("y", &float4::y)
        .def_readwrite("z", &float4::z)
        .def_readwrite("w", &float4::w)
        ;
    #endif

    py::class_<Scalar2, std::shared_ptr<Scalar2> >(m,"Scalar2")
        .def(py::init<>())
        .def_readwrite("x", &Scalar2::x)
        .def_readwrite("y", &Scalar2::y)
        ;
    py::class_<Scalar3, std::shared_ptr<Scalar3> >(m,"Scalar3")
        .def(py::init<>())
        .def_readwrite("x", &Scalar3::x)
        .def_readwrite("y", &Scalar3::y)
        .def_readwrite("z", &Scalar3::z)
        ;
    py::class_<Scalar4, std::shared_ptr<Scalar4> >(m,"Scalar4")
        .def(py::init<>())
        .def_readwrite("x", &Scalar4::x)
        .def_readwrite("y", &Scalar4::y)
        .def_readwrite("z", &Scalar4::z)
        .def_readwrite("w", &Scalar4::w)
        ;
    py::class_<uint2, std::shared_ptr<uint2> >(m,"uint2")
        .def(py::init<>())
        .def_readwrite("x", &uint2::x)
        .def_readwrite("y", &uint2::y)
        ;
    py::class_<uint3, std::shared_ptr<uint3> >(m,"uint3")
        .def(py::init<>())
        .def_readwrite("x", &uint3::x)
        .def_readwrite("y", &uint3::y)
        .def_readwrite("z", &uint3::z)
        ;
    py::class_<uint4, std::shared_ptr<uint4> >(m,"uint4")
        .def(py::init<>())
        .def_readwrite("x", &uint4::x)
        .def_readwrite("y", &uint4::y)
        .def_readwrite("z", &uint4::z)
        .def_readwrite("z", &uint4::w)
        ;
    py::class_<int2, std::shared_ptr<int2> >(m,"int2")
        .def(py::init<>())
        .def_readwrite("x", &int2::x)
        .def_readwrite("y", &int2::y)
        ;
    py::class_<int3, std::shared_ptr<int3> >(m,"int3")
        .def(py::init<>())
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        ;
    py::class_<int4, std::shared_ptr<int4> >(m,"int4")
        .def(py::init<>())
        .def_readwrite("x", &int4::x)
        .def_readwrite("y", &int4::y)
        .def_readwrite("z", &int4::z)
        .def_readwrite("z", &int4::w)
        ;
    py::class_<char3, std::shared_ptr<char3> >(m,"char3")
        .def(py::init<>())
        .def_readwrite("x", &char3::x)
        .def_readwrite("y", &char3::y)
        .def_readwrite("z", &char3::z)
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
