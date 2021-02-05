// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldClassPlane.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif


DEVICE bool ManifoldClassPlane::validate(const BoxDim box)
{
 Scalar3 lo = box.getLo();
 Scalar3 hi = box.getHi();
 if (shift > hi.z || shift < lo.z)
     {
     return true;
     }
     else return false;
}

//! Exports the Plane manifold class to python
void export_ManifoldClassPlane(pybind11::module& m)
    {
    py::class_< ManifoldClassPlane, std::shared_ptr<ManifoldClassPlane> >(m, "ManifoldClassPlane")
    .def(py::init<Scalar >())
    .def("implicit_function", &ManifoldClassPlane::implicit_function)
    .def("derivative", &ManifoldClassPlane::derivative)
    ;
    }
