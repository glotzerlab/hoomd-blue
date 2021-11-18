// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mspells

/*! \file NoFriction.h
  \brief Declares the pure virtual NoFriction class
*/

#ifndef __NOFRICTION_H__
#define __NOFRICTION_H__

#include "VectorMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#undef DEVICE
#ifdef __HIPCC__
#define DEVICE __host__ __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace dem
    {
template<typename Real> class NoFriction
    {
    public:
    NoFriction() { }

    DEVICE static bool needsVelocity()
        {
        return false;
        }

    DEVICE inline void setVelocity(const vec3<Real>& v) { }

    DEVICE inline void swapij() { }

    template<typename Vec>
    DEVICE inline Vec modifiedForce(const Vec& r0Prime, const Vec& force) const
        {
        return force;
        }
    };

    } // end namespace dem
    } // end namespace hoomd

#endif
