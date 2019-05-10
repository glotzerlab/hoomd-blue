// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __HOOMD_TEXTURE_TOOLS_H__
#define __HOOMD_TEXTURE_TOOLS_H__

/*! \file TextureTools.h
    \brief Utilities for working with textures

    TextureTools.h previously existed to aid in defining Scalar textures which may be either float or double.

    Now, it only provides a __ldg() overload for double4.
*/

#include "HOOMDMath.h"

#ifdef NVCC

//! Fetch a double4 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    double4 value from read only memory.

    \param ptr Pointer to read
*/
__device__ inline double4 __ldg(const double4 *ptr)
    {
    int4 part1 = __ldg(((int4 *)ptr));;
    int4 part2 = __ldg(((int4 *)ptr)+1);;
    return make_double4(__hiloint2double(part1.y, part1.x),
                        __hiloint2double(part1.w, part1.z),
                        __hiloint2double(part2.y, part2.x),
                        __hiloint2double(part2.w, part2.z));
    }
#endif

#endif // __HOOMD_MATH_H__
