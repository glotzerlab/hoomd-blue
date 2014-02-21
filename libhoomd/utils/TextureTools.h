/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

typedef texture<Scalar, 1, cudaReadModeElementType> scalar_tex_t;
typedef texture<Scalar2, 1, cudaReadModeElementType> scalar2_tex_t;
typedef texture<Scalar4, 1, cudaReadModeElementType> scalar4_tex_t;

//! Fetch a Scalar value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar texFetchScalar(const Scalar *ptr, texture<Scalar, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    return tex1Dfetch(tex_ref, ii);
    #endif
    }

//! Fetch a Scalar2 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar2 value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar2 texFetchScalar2(const Scalar2 *ptr, texture<Scalar2, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    return tex1Dfetch(tex_ref, ii);
    #endif
    }

//! Fetch a Scalar4 value from texture memory.
/*! This function should called whenever a CUDA kernel wants to retrieve a
    Scalar4 value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar4 texFetchScalar4(const Scalar4 *ptr, texture<Scalar4, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    return tex1Dfetch(tex_ref, ii);
    #endif
    }

#else
typedef texture<int2, 1, cudaReadModeElementType> scalar_tex_t;
typedef texture<int4, 1, cudaReadModeElementType> scalar2_tex_t;
typedef texture<int4, 1, cudaReadModeElementType> scalar4_tex_t;

//! Fetch a Scalar value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar texFetchScalar(const Scalar *ptr, texture<int2, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    int2 val = tex1Dfetch(tex_ref, ii);
    return Scalar(__hiloint2double(val.y, val.x));
    #endif
    }

//! Fetch a Scalar2 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar2 value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar2 texFetchScalar2(const Scalar2* ptr, texture<int4, 1> tex_ref, unsigned int ii)
    {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr+ii);
    #else
    int4 val = tex1Dfetch(tex_ref, ii);
    return make_scalar2(__hiloint2double(val.y, val.x),
                        __hiloint2double(val.w, val.z));
    #endif
    }

//! Fetch a Scalar4 value from texture memory.
/*! This function should be called whenever a CUDA kernel wants to retrieve a
    Scalar4 value from texture memory.

    \param ptr Pointer to bound memory
    \param tex_ref Texture in which the desired values are stored.
    \param ii Index at which to look.
*/
__device__ inline Scalar4 texFetchScalar4(const Scalar4 *ptr, texture<int4, 1> tex_ref, unsigned int ii)
    {
    unsigned int idx = 2*ii;
    #if __CUDA_ARCH__ >= 350
    int4 part1 = __ldg(((int4 *)ptr)+idx);;
    int4 part2 = __ldg(((int4 *)ptr)+idx+1);;
    #else
    int4 part1 = tex1Dfetch(tex_ref, idx);
    int4 part2 = tex1Dfetch(tex_ref, idx+1);
    #endif
    return make_scalar4(__hiloint2double(part1.y, part1.x),
                        __hiloint2double(part1.w, part1.z),
                        __hiloint2double(part2.y, part2.x),
                        __hiloint2double(part2.w, part2.z));
    }
#endif
#endif



#endif // __HOOMD_MATH_H__
