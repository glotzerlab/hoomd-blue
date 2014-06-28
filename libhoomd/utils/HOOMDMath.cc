/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file HOOMDMath.cc
    \brief Defines code needed for common math operations
 */

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "HOOMDMath.h"

#include <boost/python.hpp>
using namespace boost::python;

void export_hoomd_math_functions()
    {
    // The use of shared_ptr's for exporting CUDA vector types is a workaround
    // see http://stackoverflow.com/questions/13177573/how-to-expose-aligned-class-with-boost-python
    #ifdef SINGLE_PRECISION
    class_<double2, boost::shared_ptr<double2> >("double2", init<>())
        .def_readwrite("x", &double2::x)
        .def_readwrite("y", &double2::y)
        ;
    class_<double3, boost::shared_ptr<double3> >("double3", init<>())
        .def_readwrite("x", &double3::x)
        .def_readwrite("y", &double3::y)
        .def_readwrite("z", &double3::z)
        ;
    class_<double4, boost::shared_ptr<double4> >("double4", init<>())
        .def_readwrite("x", &double4::x)
        .def_readwrite("y", &double4::y)
        .def_readwrite("z", &double4::z)
        .def_readwrite("w", &double4::w)
        ;
    #else
    class_<float2, boost::shared_ptr<float2> >("float2", init<>())
        .def_readwrite("x", &float2::x)
        .def_readwrite("y", &float2::y)
        ;
    class_<float3, boost::shared_ptr<float3> >("float3", init<>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        ;
    class_<float4, boost::shared_ptr<float4> >("float4", init<>())
        .def_readwrite("x", &float4::x)
        .def_readwrite("y", &float4::y)
        .def_readwrite("z", &float4::z)
        .def_readwrite("w", &float4::w)
        ;
    #endif

    class_<Scalar2, boost::shared_ptr<Scalar2> >("Scalar2", init<>())
        .def_readwrite("x", &Scalar2::x)
        .def_readwrite("y", &Scalar2::y)
        ;
    class_<Scalar3, boost::shared_ptr<Scalar3> >("Scalar3", init<>())
        .def_readwrite("x", &Scalar3::x)
        .def_readwrite("y", &Scalar3::y)
        .def_readwrite("z", &Scalar3::z)
        ;
    class_<Scalar4, boost::shared_ptr<Scalar4> >("Scalar4", init<>())
        .def_readwrite("x", &Scalar4::x)
        .def_readwrite("y", &Scalar4::y)
        .def_readwrite("z", &Scalar4::z)
        .def_readwrite("w", &Scalar4::w)
        ;
    class_<uint2, boost::shared_ptr<uint2> >("uint2", init<>())
        .def_readwrite("x", &uint2::x)
        .def_readwrite("y", &uint2::y)
        ;
    class_<uint3, boost::shared_ptr<uint3> >("uint3", init<>())
        .def_readwrite("x", &uint3::x)
        .def_readwrite("y", &uint3::y)
        .def_readwrite("z", &uint3::z)
        ;
    class_<uint4, boost::shared_ptr<uint4> >("uint4", init<>())
        .def_readwrite("x", &uint4::x)
        .def_readwrite("y", &uint4::y)
        .def_readwrite("z", &uint4::z)
        .def_readwrite("z", &uint4::w)
        ;
    class_<int2, boost::shared_ptr<int2> >("int2", init<>())
        .def_readwrite("x", &int2::x)
        .def_readwrite("y", &int2::y)
        ;
    class_<int3, boost::shared_ptr<int3> >("int3", init<>())
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        ;
    class_<int4, boost::shared_ptr<int4> >("int4", init<>())
        .def_readwrite("x", &int4::x)
        .def_readwrite("y", &int4::y)
        .def_readwrite("z", &int4::z)
        .def_readwrite("z", &int4::w)
        ;
    class_<char3, boost::shared_ptr<char3> >("char3", init<>())
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

#ifdef WIN32
#pragma warning( pop )
#endif
