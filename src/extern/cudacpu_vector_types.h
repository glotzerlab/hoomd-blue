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

#if !defined(__VECTOR_TYPES_H__)
#define __VECTOR_TYPES_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cudacpu_host_defines.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*DEVICE_BUILTIN*/
struct char1
{
  signed char x;
};

/*DEVICE_BUILTIN*/
struct uchar1 
{
  unsigned char x;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(2) char2
{
  signed char x, y;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(2) uchar2
{
  unsigned char x, y;
};

/*DEVICE_BUILTIN*/
struct char3
{
  signed char x, y, z;
};

/*DEVICE_BUILTIN*/
struct uchar3
{
  unsigned char x, y, z;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(4) char4
{
  signed char x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(4) uchar4
{
  unsigned char x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct short1
{
  short x;
};

/*DEVICE_BUILTIN*/
struct ushort1
{
  unsigned short x;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(4) short2
{
  short x, y;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(4) ushort2
{
  unsigned short x, y;
};

/*DEVICE_BUILTIN*/
struct short3
{
  short x, y, z;
};

/*DEVICE_BUILTIN*/
struct ushort3
{
  unsigned short x, y, z;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(8) short4
{
  short x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(8) ushort4
{
  unsigned short x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct int1
{
  int x;
};

/*DEVICE_BUILTIN*/
struct uint1
{
  unsigned int x;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(8) int2
{
  int x, y;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(8) uint2
{
  unsigned int x, y;
};

/*DEVICE_BUILTIN*/
struct int3
{
  int x, y, z;
};

/*DEVICE_BUILTIN*/
struct uint3
{
  unsigned int x, y, z;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) int4
{
  int x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) uint4
{
  unsigned int x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct long1
{
  long int x;
};

/*DEVICE_BUILTIN*/
struct ulong1
{
  unsigned long x;
};

/*DEVICE_BUILTIN*/
struct 
#if defined (_WIN32)
       __builtin_align__(8)
#else /* _WIN32 */
       __builtin_align__(2*sizeof(long int))
#endif /* _WIN32 */
                                             long2
{
  long int x, y;
};

/*DEVICE_BUILTIN*/
struct 
#if defined (_WIN32)
       __builtin_align__(8)
#else /* _WIN32 */
       __builtin_align__(2*sizeof(unsigned long int))
#endif /* _WIN32 */
                                                      ulong2
{
  unsigned long int x, y;
};

#if !defined(__LP64__)

/*DEVICE_BUILTIN*/
struct long3
{
  long int x, y, z;
};

/*DEVICE_BUILTIN*/
struct ulong3
{
  unsigned long int x, y, z;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) long4
{
  long int x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) ulong4
{
  unsigned long int x, y, z, w;
};

#endif /* !__LP64__ */

/*DEVICE_BUILTIN*/
struct float1
{
  float x;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(8) float2
{
  float x, y;
};

/*DEVICE_BUILTIN*/
struct float3
{
  float x, y, z;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) float4
{
  float x, y, z, w;
};

/*DEVICE_BUILTIN*/
struct double1
{
  double x;
};

/*DEVICE_BUILTIN*/
struct __builtin_align__(16) double2
{
  double x, y;
};

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*DEVICE_BUILTIN*/
typedef struct char1 char1;
/*DEVICE_BUILTIN*/
typedef struct uchar1 uchar1;
/*DEVICE_BUILTIN*/
typedef struct char2 char2;
/*DEVICE_BUILTIN*/
typedef struct uchar2 uchar2;
/*DEVICE_BUILTIN*/
typedef struct char3 char3;
/*DEVICE_BUILTIN*/
typedef struct uchar3 uchar3;
/*DEVICE_BUILTIN*/
typedef struct char4 char4;
/*DEVICE_BUILTIN*/
typedef struct uchar4 uchar4;
/*DEVICE_BUILTIN*/
typedef struct short1 short1;
/*DEVICE_BUILTIN*/
typedef struct ushort1 ushort1;
/*DEVICE_BUILTIN*/
typedef struct short2 short2;
/*DEVICE_BUILTIN*/
typedef struct ushort2 ushort2;
/*DEVICE_BUILTIN*/
typedef struct short3 short3;
/*DEVICE_BUILTIN*/
typedef struct ushort3 ushort3;
/*DEVICE_BUILTIN*/
typedef struct short4 short4;
/*DEVICE_BUILTIN*/
typedef struct ushort4 ushort4;
/*DEVICE_BUILTIN*/
typedef struct int1 int1;
/*DEVICE_BUILTIN*/
typedef struct uint1 uint1;
/*DEVICE_BUILTIN*/
typedef struct int2 int2;
/*DEVICE_BUILTIN*/
typedef struct uint2 uint2;
/*DEVICE_BUILTIN*/
typedef struct int3 int3;
/*DEVICE_BUILTIN*/
typedef struct uint3 uint3;
/*DEVICE_BUILTIN*/
typedef struct int4 int4;
/*DEVICE_BUILTIN*/
typedef struct uint4 uint4;
/*DEVICE_BUILTIN*/
typedef struct long1 long1;
/*DEVICE_BUILTIN*/
typedef struct ulong1 ulong1;
/*DEVICE_BUILTIN*/
typedef struct long2 long2;
/*DEVICE_BUILTIN*/
typedef struct ulong2 ulong2;
/*DEVICE_BUILTIN*/
typedef struct long3 long3;
/*DEVICE_BUILTIN*/
typedef struct ulong3 ulong3;
/*DEVICE_BUILTIN*/
typedef struct long4 long4;
/*DEVICE_BUILTIN*/
typedef struct ulong4 ulong4;
/*DEVICE_BUILTIN*/
typedef struct float1 float1;
/*DEVICE_BUILTIN*/
typedef struct float2 float2;
/*DEVICE_BUILTIN*/
typedef struct float3 float3;
/*DEVICE_BUILTIN*/
typedef struct float4 float4;
/*DEVICE_BUILTIN*/
typedef struct double1 double1;
/*DEVICE_BUILTIN*/
typedef struct double2 double2;

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/*DEVICE_BUILTIN*/
typedef struct dim3 dim3;

/*DEVICE_BUILTIN*/
struct dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif /* __cplusplus */
};

#endif /* !__VECTOR_TYPES_H__ */
