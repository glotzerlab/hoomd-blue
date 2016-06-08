// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file HOOMDMath.cc
    \brief Defines code needed for common math operations
 */


#include "HOOMDMath.h"

#include <boost/python.hpp>
using namespace boost::python;

void export_hoomd_math_functions()
    {
    // The use of shared_ptr's for exporting CUDA vector types is a workaround
    // see http://stackoverflow.com/questions/13177573/how-to-expose-aligned-class-with-boost-python
    #ifdef SINGLE_PRECISION
    class_<double2, std::shared_ptr<double2> >("double2", init<>())
        .def_readwrite("x", &double2::x)
        .def_readwrite("y", &double2::y)
        ;
    class_<double3, std::shared_ptr<double3> >("double3", init<>())
        .def_readwrite("x", &double3::x)
        .def_readwrite("y", &double3::y)
        .def_readwrite("z", &double3::z)
        ;
    class_<double4, std::shared_ptr<double4> >("double4", init<>())
        .def_readwrite("x", &double4::x)
        .def_readwrite("y", &double4::y)
        .def_readwrite("z", &double4::z)
        .def_readwrite("w", &double4::w)
        ;
    #else
    class_<float2, std::shared_ptr<float2> >("float2", init<>())
        .def_readwrite("x", &float2::x)
        .def_readwrite("y", &float2::y)
        ;
    class_<float3, std::shared_ptr<float3> >("float3", init<>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        ;
    class_<float4, std::shared_ptr<float4> >("float4", init<>())
        .def_readwrite("x", &float4::x)
        .def_readwrite("y", &float4::y)
        .def_readwrite("z", &float4::z)
        .def_readwrite("w", &float4::w)
        ;
    #endif

    class_<Scalar2, std::shared_ptr<Scalar2> >("Scalar2", init<>())
        .def_readwrite("x", &Scalar2::x)
        .def_readwrite("y", &Scalar2::y)
        ;
    class_<Scalar3, std::shared_ptr<Scalar3> >("Scalar3", init<>())
        .def_readwrite("x", &Scalar3::x)
        .def_readwrite("y", &Scalar3::y)
        .def_readwrite("z", &Scalar3::z)
        ;
    class_<Scalar4, std::shared_ptr<Scalar4> >("Scalar4", init<>())
        .def_readwrite("x", &Scalar4::x)
        .def_readwrite("y", &Scalar4::y)
        .def_readwrite("z", &Scalar4::z)
        .def_readwrite("w", &Scalar4::w)
        ;
    class_<uint2, std::shared_ptr<uint2> >("uint2", init<>())
        .def_readwrite("x", &uint2::x)
        .def_readwrite("y", &uint2::y)
        ;
    class_<uint3, std::shared_ptr<uint3> >("uint3", init<>())
        .def_readwrite("x", &uint3::x)
        .def_readwrite("y", &uint3::y)
        .def_readwrite("z", &uint3::z)
        ;
    class_<uint4, std::shared_ptr<uint4> >("uint4", init<>())
        .def_readwrite("x", &uint4::x)
        .def_readwrite("y", &uint4::y)
        .def_readwrite("z", &uint4::z)
        .def_readwrite("z", &uint4::w)
        ;
    class_<int2, std::shared_ptr<int2> >("int2", init<>())
        .def_readwrite("x", &int2::x)
        .def_readwrite("y", &int2::y)
        ;
    class_<int3, std::shared_ptr<int3> >("int3", init<>())
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        ;
    class_<int4, std::shared_ptr<int4> >("int4", init<>())
        .def_readwrite("x", &int4::x)
        .def_readwrite("y", &int4::y)
        .def_readwrite("z", &int4::z)
        .def_readwrite("z", &int4::w)
        ;
    class_<char3, std::shared_ptr<char3> >("char3", init<>())
        .def_readwrite("x", &char3::x)
        .def_readwrite("y", &char3::y)
        .def_readwrite("z", &char3::z)
        ;

    def("make_scalar2", &make_scalar2);
    def("make_scalar3", &make_scalar3);
    def("make_scalar4", &make_scalar4);
    def("make_uint2", &make_uint2);
    def("make_uint3", &make_uint3);
    def("make_uint4", &make_uint4);
    def("make_int2", &make_int2);
    def("make_int3", &make_int3);
    def("make_int4", &make_int4);
    def("make_char3", &make_char3);
    def("int_as_scalar", &__int_as_scalar);
    }
