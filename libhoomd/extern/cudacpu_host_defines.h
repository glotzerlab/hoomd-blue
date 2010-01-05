/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#if !defined(__HOST_DEFINES_H__)
#define __HOST_DEFINES_H__

#if !defined(__GNUC__) && !defined(_WIN32)

#error --- !!! UNSUPPORTED COMPILER !!! ---

#elif defined(__GNUC__)

#define __no_return__ \
        __attribute__((__noreturn__))
#define __noinline__ \
        __attribute__((__noinline__))
#define __align__(n) \
        __attribute__((__aligned__(n)))
#define __thread__ \
        __thread
#define __import__
#define __export__
#define __location__(a) \
        __loc__(__attribute__((a)))
#define CUDARTAPI

#elif defined(_WIN32)

#if _MSC_VER >= 1400

#define __restrict__ \
        __restrict

#else /* _MSC_VER >= 1400 */

#define __restrict__

#endif /* _MSC_VER >= 1400 */

#define __inline__ \
        __inline
#define __no_return__ \
        __declspec(noreturn)
#define __noinline__ \
        __declspec(noinline)
#define __align__(n) \
        __declspec(align(n))
#define __thread__ \
        __declspec(thread)
#define __import__ \
        __declspec(dllimport)
#define __export__ \
        __declspec(dllexport)
#define __location__(a) \
        __loc__(__declspec(a))
#define CUDARTAPI \
        __stdcall

#endif /* !__GNUC__ && !_WIN32 */

#if defined(__CUDACC__) || defined(__CUDABE__)

#define __loc__(a) \
        a
#define __builtin_align__(a) \
        __align__(a)

#else /* __CUDACC__ || __CUDABE__ */

#define __loc__(a)
#define __builtin_align__(a)

#endif /* __CUDACC__ || __CUDABE__ */

#define __device__ \
        __location__(__device__)
#define __host__ \
        __location__(__host__)
#define __global__ \
        __location__(__global__)
#define __shared__ \
        __location__(__shared__)
#define __constant__ \
        __location__(__constant__)
#define __launch_bounds__(t, b) \
        __location__(__launch_bounds__(t, b))

#endif /* !__HOST_DEFINES_H__ */
