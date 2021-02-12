// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldClassPrimitive.h"

namespace py = pybind11;

using namespace std;

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

DEVICE bool ManifoldClassPrimitive::validate(const BoxDim box)
{
 Scalar3 box_length = box.getHi() - box.getLo();

 Lx = 2*M_PI*Nx/box_length.x;
 Ly = 2*M_PI*Ny/box_length.y;
 Lz = 2*M_PI*Nz/box_length.z;

 return false;
}

//! Exports the Primitive manifold class to python
void export_ManifoldClassPrimitive(pybind11::module& m)
    {
    py::class_< ManifoldClassPrimitive, std::shared_ptr<ManifoldClassPrimitive> >(m, "ManifoldClassPrimitive")
    .def(py::init<int, int, int, Scalar >())
    .def("implicit_function", &ManifoldClassPrimitive::implicit_function)
    .def("derivative", &ManifoldClassPrimitive::derivative)
    ;
    }

