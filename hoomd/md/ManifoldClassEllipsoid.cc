// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldClassEllipsoid.h"
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

DEVICE bool ManifoldClassEllipsoid::validate(const BoxDim box)
{
 Scalar3 lo = box.getLo();
 Scalar3 hi = box.getHi();
 Scalar ia = Scalar(1.0)/fast::sqrt(a);
 Scalar ib = Scalar(1.0)/fast::sqrt(b);
 Scalar ic = Scalar(1.0)/fast::sqrt(c);

 if (Px + ia > hi.x || Px - ia < lo.x ||
     Py + ib > hi.y || Py - ib < lo.y ||
     Pz + ic > hi.z || Pz - ic < lo.z)
     {
     return true;
     }
     else return false;
}


//! Exports the Ellipsoid manifold class to python
void export_ManifoldClassEllipsoid(pybind11::module& m)
    {
    py::class_< ManifoldClassEllipsoid, std::shared_ptr<ManifoldClassEllipsoid> >(m, "ManifoldClassEllipsoid")
    .def(py::init<Scalar, Scalar, Scalar, Scalar3 >())
    .def("implicit_function", &ManifoldClassEllipsoid::implicit_function)
    .def("derivative", &ManifoldClassEllipsoid::derivative)
    ;
    }


