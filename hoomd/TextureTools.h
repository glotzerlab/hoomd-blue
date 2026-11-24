// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __HOOMD_TEXTURE_TOOLS_H__
#define __HOOMD_TEXTURE_TOOLS_H__

/*! \file TextureTools.h
    \brief Utilities for working with textures

    TextureTools.h previously existed to aid in defining Scalar textures which may be either float
   or double.

    Now, it only provides a __ldg() overload for double4.
*/

#include "HOOMDMath.h"

#ifdef __NVCC__

#if CUDART_VERSION >= 13000
#define DOUBLE4 double4_16a
#else
#define DOUBLE4 double4
#endif

//! Fetch a double4 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    double4 value from read only memory.

    \param ptr Pointer to read
*/
__device__ inline DOUBLE4 __ldg(const DOUBLE4* ptr)
    {
    int4 part1 = __ldg(((int4*)ptr));
    int4 part2 = __ldg(((int4*)ptr) + 1);
    return DOUBLE4 {__hiloint2double(part1.y, part1.x),
                    __hiloint2double(part1.w, part1.z),
                    __hiloint2double(part2.y, part2.x),
                    __hiloint2double(part2.w, part2.z)};
    }

#undef DOUBLE4

#endif // __NVCC__

#endif // __HOOMD_TEXTURE_TOOLS_H__
