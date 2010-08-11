/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifndef _PARTICLEDATA_CUH_
#define _PARTICLEDATA_CUH_

#include <cuda_runtime.h>

/*! \file ParticleData.cuh
    \brief Declares GPU kernel code and data structure functions used by ParticleData
*/

#ifdef NVCC
//! Sentinel value in \a body to signify that this particle does not belong to a rigid body
const unsigned int NO_BODY = 0xffffffff;
#endif

//! Structure of arrays of the particle data as it resides on the GPU
/*! Stores pointers to the particles positions, velocities, acceleartions, and particle tags.
    Particle type information is most likely needed along with the position, so the type
    is encoded in the 4th float in the position float4 as an integer. Device code
    can decode this type data with __float_as_int();

    All the pointers in this structure are allocated on the device.

    This structure is about to be rewritten. Consider it being documented as poorly documented
    for now.

    \ingroup gpu_data_structs
*/
struct gpu_pdata_arrays
    {
    unsigned int N;         //!< Number of particles in the arrays
    
    float4 *pos;        //!< Particle position in \c x,\c y,\c z, particle type as an int in \c w
    float4 *vel;        //!< Particle velocity in \c x, \c y, \c z, nothing in \c w
    float4 *accel;      //!< Particle acceleration in \c x, \c y, \c z, nothing in \c w
    float *charge;      //!< Particle charge
    float *mass;        //!< Particle mass
    float *diameter;    //!< Particle diameter
    int4 *image;        //!< Particle box image location in \c x, c y, and \c z. Nothing in \c w.
    
    unsigned int *tag;  //!< Particle tag
    unsigned int *rtag; //!< Particle rtag
    unsigned int *body; //!< Particle rigid body (0xffffffff if not in a body)
    };

//! Store the box size on the GPU
/*! \note For performance reasons, the GPU code is allowed to assume that the box goes
    from -L/2 to L/2, and the box dimensions in this structure must reflect that.

    \ingroup gpu_data_structs
*/
struct gpu_boxsize
    {
    float Lx;   //!< Length of the box in the x-direction
    float Ly;   //!< Length of the box in the y-direction
    float Lz;   //!< Length of teh box in the z-direction
    float Lxinv;//!< 1.0f/Lx
    float Lyinv;//!< 1.0f/Ly
    float Lzinv;//!< 1.0f/Lz
    };

//! Helper kernel for un-interleaving data
cudaError_t gpu_uninterleave_float4(float *d_out, float4 *d_in, int N, int pitch);
//! Helper kernel for interleaving data
cudaError_t gpu_interleave_float4(float4 *d_out, float *d_in, int N, int pitch);

//! Generate a test pattern in the data on the GPU (for unit testing)
cudaError_t gpu_generate_pdata_test(const gpu_pdata_arrays &pdata);
//! Read from the pos texture and write into the vel array (for unit testing)
cudaError_t gpu_pdata_texread_test(const gpu_pdata_arrays &pdata);

#endif

