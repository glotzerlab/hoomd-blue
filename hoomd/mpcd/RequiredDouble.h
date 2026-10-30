// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_REQUIRED_DOUBLE_H_
#define MPCD_REQUIRED_DOUBLE_H_

/*
 * This file introduces a compatibility typedef for newer versions of CUDA to
 * require the use of a RequiredDouble4 data type.
 *
 * The intention is for this to be a temporary fix until a more involved
 * refactoring of the mixed-precision model in MPCD can be carried out.
 */

#include "hoomd/HOOMDMath.h"

#ifdef __NVCC__
#define HOSTDEVICE __forceinline__ __host__ __device__
#else
#define HOSTDEVICE inline
#endif

namespace hoomd
    {
namespace mpcd
    {
#if !defined(ENABLE_GPU) || (defined(ENABLE_GPU) && CUDART_VERSION >= 13000)
typedef double4_16a RequiredDouble4;
#else
typedef double4 RequiredDouble4;
#endif

HOSTDEVICE RequiredDouble4 make_RequiredDouble4(double x, double y, double z, double w)
    {
    return RequiredDouble4 {x, y, z, w};
    }

    } // end namespace mpcd
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // MPCD_REQUIRED_DOUBLE_H_
