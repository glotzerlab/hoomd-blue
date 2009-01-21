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

#if !defined(__VECTOR_FUNCTIONS_H__)
#define __VECTOR_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cudacpu_host_defines.h"
#include "cudacpu_vector_types.h"

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

static __inline__ __host__ __device__ char1 make_char1(signed char x)
{
  char1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ uchar1 make_uchar1(unsigned char x)
{
  uchar1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ char2 make_char2(signed char x, signed char y)
{
  char2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ short1 make_short1(short x)
{
  short1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ ushort1 make_ushort1(unsigned short x)
{
  ushort1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ short2 make_short2(short x, short y)
{
  short2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ ushort2 make_ushort2(unsigned short x, unsigned short y)
{
  ushort2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ short3 make_short3(short x,short y, short z)
{ 
  short3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
  ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ short4 make_short4(short x, short y, short z, short w)
{
  short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
  ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ int1 make_int1(int x)
{
  int1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ uint1 make_uint1(unsigned int x)
{
  uint1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ int2 make_int2(int x, int y)
{
  int2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ int3 make_int3(int x, int y, int z)
{
  int3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ long1 make_long1(long int x)
{
  long1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ long2 make_long2(long int x, long int y)
{
  long2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t; t.x = x; t.y = y; return t;
}

#if !defined(__LP64__)

static __inline__ __host__ __device__ long3 make_long3(long int x, long int y, long int z)
{
  long3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

#endif /* !__LP64__ */

static __inline__ __host__ __device__ float1 make_float1(float x)
{
  float1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ float2 make_float2(float x, float y)
{
  float2 t; t.x = x; t.y = y; return t;
}

static __inline__ __host__ __device__ float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

static __inline__ __host__ __device__ float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static __inline__ __host__ __device__ double1 make_double1(double x)
{
  double1 t; t.x = x; return t;
}

static __inline__ __host__ __device__ double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

#endif /* !__VECTOR_FUNCTIONS_H__ */
