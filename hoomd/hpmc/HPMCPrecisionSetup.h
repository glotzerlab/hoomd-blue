// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"

/*! \file HPMCPrecisionSetup.h
    \brief Setup for hpmc mixed precision
*/

#ifndef __HPMC_PRECISION_SETUP_H__
#define __HPMC_PRECISION_SETUP_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace hpmc
    {
typedef ShortReal OverlapReal;
typedef ShortReal3 OverlapReal3;
typedef ShortReal4 OverlapReal4;

//! Helper function to create OverlapReal3's from python
DEVICE inline OverlapReal3 make_overlapreal3(OverlapReal x, OverlapReal y, OverlapReal z)
    {
    OverlapReal3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
    }

//! Helper function to create OverlapReal3's from python
DEVICE inline OverlapReal4
make_overlapreal4(OverlapReal x, OverlapReal y, OverlapReal z, OverlapReal w)
    {
    OverlapReal4 result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
    }

    } // end namespace hpmc
    } // end namespace hoomd
#undef DEVICE

#endif //__HPMC_PRECISION_SETUP_H__
