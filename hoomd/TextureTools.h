// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __HOOMD_TEXTURE_TOOLS_H__
#define __HOOMD_TEXTURE_TOOLS_H__

/*! \file TextureTools.h
    \brief Utilities for working with textures

    TextureTools.h exists to aid in defining Scalar textures which may be either float or double. It aims to simplify
    code that reads from these textures so that the amount of conditional code is simplified to be entirely within
    this header.

    Planning for the future (__ldg), the fetch methods will also take in a pointer to the memory. That way, the initial
    work done to convert the texture loads over to the single/double will also make it easy to change over to __ldg
    in a single spot.
*/

#include "HOOMDMath.h"

#ifdef NVCC

//! Fetch an unsigned int from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    unsigned int value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline unsigned int texFetchUint(const unsigned int *ptr, texture<unsigned int, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    return tex1Dfetch(tex_ref, ii);
    #endif
    }

#ifdef SINGLE_PRECISION

//! Fetch a Scalar4 value from texture memory.
/*! This function should called whenever a CUDA kernel wants to retrieve a
    Scalar4 value from texture memory.

    \param ptr Pointer to bound memory
    \param ii Index at which to look.
*/
__device__ inline Scalar4 texFetchScalar4(const Scalar4 *ptr, unsigned int ii)
    {
    return __ldg(ptr+ii);
    }

#else

//! Fetch a Scalar4 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar4 value from texture memory.

    \param ptr Pointer to bound memory
    \param ii Index at which to look.
*/
__device__ inline Scalar4 texFetchScalar4(const Scalar4 *ptr, unsigned int ii)
    {
    unsigned int idx = 2*ii;
    int4 part1 = __ldg(((int4 *)ptr)+idx);;
    int4 part2 = __ldg(((int4 *)ptr)+idx+1);;
    return make_scalar4(__hiloint2double(part1.y, part1.x),
                        __hiloint2double(part1.w, part1.z),
                        __hiloint2double(part2.y, part2.x),
                        __hiloint2double(part2.w, part2.z));
    }
#endif
#endif

#endif // __HOOMD_MATH_H__
