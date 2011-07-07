/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
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
    class_<Scalar2>("Scalar2", init<>())
        .def_readwrite("x", &Scalar2::x)
        .def_readwrite("y", &Scalar2::y)
        ;
    class_<Scalar3>("Scalar3", init<>())
        .def_readwrite("x", &Scalar3::x)
        .def_readwrite("y", &Scalar3::y)
        .def_readwrite("z", &Scalar3::z)
        ;
    class_<Scalar4>("Scalar4", init<>())
        .def_readwrite("x", &Scalar4::x)
        .def_readwrite("y", &Scalar4::y)
        .def_readwrite("z", &Scalar4::z)
        .def_readwrite("w", &Scalar4::w)
        ;
    class_<uint2>("uint2", init<>())
        .def_readwrite("x", &uint2::x)
        .def_readwrite("y", &uint2::y)
        ;
    class_<uint3>("uint3", init<>())
        .def_readwrite("x", &uint3::x)
        .def_readwrite("y", &uint3::y)
        .def_readwrite("z", &uint3::z)
        ;
    class_<uint4>("uint4", init<>())
        .def_readwrite("x", &uint4::x)
        .def_readwrite("y", &uint4::y)
        .def_readwrite("z", &uint4::z)
        .def_readwrite("z", &uint4::w)
        ;
    class_<int2>("int2", init<>())
        .def_readwrite("x", &int2::x)
        .def_readwrite("y", &int2::y)
        ;
    class_<int3>("int3", init<>())
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        ;
    class_<int4>("int4", init<>())
        .def_readwrite("x", &int4::x)
        .def_readwrite("y", &int4::y)
        .def_readwrite("z", &int4::z)
        .def_readwrite("z", &int4::w)
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
    }

#ifdef WIN32
#pragma warning( pop )
#endif

