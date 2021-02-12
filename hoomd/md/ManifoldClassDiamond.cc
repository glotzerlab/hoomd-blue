// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldClassDiamond.h"
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

DEVICE bool ManifoldClassDiamond::validate(const BoxDim box)
{
 Scalar3 box_length = box.getHi() - box.getLo();

 Lx = M_PI*Nx/box_length.x;
 Ly = M_PI*Ny/box_length.y;
 Lz = M_PI*Nz/box_length.z;

 return false;
}

//! Exports the Diamond manifold class to python
void export_ManifoldClassDiamond(pybind11::module& m)
    {
    py::class_< ManifoldClassDiamond, std::shared_ptr<ManifoldClassDiamond> >(m, "ManifoldClassDiamond")
    .def(py::init<int, int, int, Scalar >())
    .def("implicit_function", &ManifoldClassDiamond::implicit_function)
    .def("derivative", &ManifoldClassDiamond::derivative)
    ;
    }

