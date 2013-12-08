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

#ifndef _PARTICLEDATA_CUH_
#define _PARTICLEDATA_CUH_

#include <cuda_runtime.h>
#include "BoxDim.h"

#include "cached_allocator.h"
#include "moderngpu/util/mgpucontext.h"

/*! \file ParticleData.cuh
    \brief Declares GPU kernel code and data structure functions used by ParticleData
*/

#ifdef NVCC
//! Sentinel value in \a body to signify that this particle does not belong to a rigid body
const unsigned int NO_BODY = 0xffffffff;
//! Sentinel value in \a r_tag to signify that this particle is not currently present on the local processor
const unsigned int NOT_LOCAL = 0xffffffff;

#ifdef ENABLE_MPI
//! Sentinel value in \a r_tag to signify that the particle is to be removed from the local processor
const unsigned int STAGED = 0xfffffffe;
#endif

#endif

#ifdef NVCC
//! Compact particle data storage
struct pdata_element
    {
    Scalar4 pos;               //!< Position
    Scalar4 vel;               //!< Velocity
    Scalar3 accel;             //!< Acceleration
    Scalar charge;             //!< Charge
    Scalar diameter;           //!< Diameter
    int3 image;                //!< Image
    unsigned int body;         //!< Body id
    Scalar4 orientation;       //!< Orientation
    unsigned int tag;          //!< global tag
    };
#else
//!Forward declaration
class pdata_element;
#endif

//! Pack particle data into output buffer and remove marked particles
unsigned int gpu_pdata_remove(const unsigned int N,
                    const Scalar4 *d_pos,
                    const Scalar4 *d_vel,
                    const Scalar3 *d_accel,
                    const Scalar *d_charge,
                    const Scalar *d_diameter,
                    const int3 *d_image,
                    const unsigned int *d_body,
                    const Scalar4 *d_orientation,
                    const unsigned int *d_tag,
                    unsigned int *d_rtag,
                    Scalar4 *d_pos_alt,
                    Scalar4 *d_vel_alt,
                    Scalar3 *d_accel_alt,
                    Scalar *d_charge_alt,
                    Scalar *d_diameter_alt,
                    int3 *d_image_alt,
                    unsigned int *d_body_alt,
                    Scalar4 *d_orientation_alt,
                    unsigned int *d_tag_alt,
                    pdata_element *d_out,
                    unsigned int *d_comm_flags,
                    unsigned int *d_comm_flags_out,
                    unsigned int max_n_out,
                    mgpu::ContextPtr mgpu_context,
                    cached_allocator& alloc);

//! Update reverse-lookup tabs
void gpu_pdata_update_rtags(
    const unsigned int *d_tag,
    unsigned int *d_rtag,
    const unsigned int N,
    cached_allocator& alloc);

//! Update particle data with new particles
void gpu_pdata_add_particles(const unsigned int old_nparticles,
                    const unsigned int num_add_ptls,
                    Scalar4 *d_pos,
                    Scalar4 *d_vel,
                    Scalar3 *d_accel,
                    Scalar *d_charge,
                    Scalar *d_diameter,
                    int3 *d_image,
                    unsigned int *d_body,
                    Scalar4 *d_orientation,
                    unsigned int *d_tag,
                    unsigned int *d_rtag,
                    const pdata_element *d_in,
                    unsigned int *d_comm_flags,
                    cached_allocator& alloc);
#endif
