// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "ManifoldClassSphere.h"
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

//! Evaluate deriviative of implicit function
/*! \param point Point at surface is calculated

    \return normal of the Sphere surface at input point
*/

DEVICE bool ManifoldClassSphere::validate(const BoxDim box)
{
 Scalar3 lo = box.getLo();
 Scalar3 hi = box.getHi();
 Scalar sqR = fast::sqrt(R);
 if (Px + sqR > hi.x || Px - sqR < lo.x ||
     Py + sqR > hi.y || Py - sqR < lo.y ||
     Pz + sqR > hi.z || Pz - sqR < lo.z)
     {
     return true;
     }
     else return false;
}

//! Exports the Sphere manifold class to python
void export_ManifoldClassSphere(pybind11::module& m)
    {
    py::class_< ManifoldClassSphere, std::shared_ptr<ManifoldClassSphere> >(m, "ManifoldClassSphere")
    .def(py::init<Scalar, Scalar3 >())
    .def("implicit_function", &ManifoldClassSphere::implicit_function)
    .def("derivative", &ManifoldClassSphere::derivative)
    ;
    }
