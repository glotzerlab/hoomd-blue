// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

#include <hoomd/HOOMDMath.h>
#include <hoomd/VectorMath.h>

#ifndef __DEM_VECTOR_MATH_H__
#define __DEM_VECTOR_MATH_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#undef DEVICE
#ifdef NVCC
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

template<typename Real>
DEVICE vec2<Real> vec_from_scalar2(const Scalar2 &v)
    {
    return vec2<Real>(v.x, v.y);
    }

//! z-component of the cross product of two vec2s
/*! \param a First vector
  \param b Second vector

  Returns the cross product a.x * b.y - a.y * b.x.
*/
template < class Real >
DEVICE inline Real cross(const vec2<Real>& a, const vec2<Real>& b)
    {
    return a.x * b.y - a.y * b.x;
    }

#endif //__DEM_VECTOR_MATH_H__
