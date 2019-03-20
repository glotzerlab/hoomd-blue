// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"

/*! \file HPMCPrecisionSetup.h
    \brief Setup for hpmc mixed precision
*/

#ifndef __HPMC_PRECISION_SETUP_H__
#define __HPMC_PRECISION_SETUP_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hpmc
{

#ifdef SINGLE_PRECISION

// in single precision, OverlapReal is always float
//! Typedef'd real for use in local overlap checks
typedef float  OverlapReal;
//! Typedef'd real3 for use in the local overlap checks
typedef float3 OverlapReal3;
typedef float4 OverlapReal4;

#else

// in double precision, mixed mode enables floats for OverlapReal, otherwise it is double
#ifdef ENABLE_HPMC_MIXED_PRECISION
typedef float OverlapReal;
typedef float3 OverlapReal3;
typedef float4 OverlapReal4;

#else
typedef double OverlapReal;
typedef double3 OverlapReal3;
typedef double4 OverlapReal4;

#endif

#endif

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
DEVICE inline OverlapReal4 make_overlapreal4(OverlapReal x, OverlapReal y, OverlapReal z, OverlapReal w)
    {
    OverlapReal4 result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.w = w;
    return result;
    }

}; // end namespace hpmc

#undef DEVICE

#endif //__HPMC_PRECISION_SETUP_H__
