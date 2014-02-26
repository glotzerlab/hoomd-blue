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

// Maintainer: jglaser

#ifndef __SFC_PACK_UPDATER_GPU_CUH__
#define __SFC_PACK_UPDATER_GPU_CUH__

#include "HOOMDMath.h"
#include "BoxDim.h"

#include "util/mgpucontext.h"

/*! \file SFCPackUpdaterGPU.cuh
    \brief Defines GPU functions for generating the space-filling curve sorted order on the GPU. Used by SFCPackUpdaterGPU.
*/

//! Generate sorted order on GPU
void gpu_generate_sorted_order(unsigned int N,
        const Scalar4 *d_pos,
        unsigned int *d_particle_bins,
        unsigned int *d_traversal_order,
        unsigned int n_grid,
        unsigned int *d_sorted_order,
        const BoxDim& box,
        bool twod,
        mgpu::ContextPtr mgpu_context);

//! Reorder particle data (GPU driver function)
void gpu_apply_sorted_order(
        unsigned int N,
        const unsigned int *d_sorted_order,
        const Scalar4 *d_pos,
        Scalar4 *d_pos_alt,
        const Scalar4 *d_vel,
        Scalar4 *d_vel_alt,
        const Scalar3 *d_accel,
        Scalar3 *d_accel_alt,
        const Scalar *d_charge,
        Scalar *d_charge_alt,
        const Scalar *d_diameter,
        Scalar *d_diameter_alt,
        const int3 *d_image,
        int3 *d_image_alt,
        const unsigned int *d_body,
        unsigned int *d_body_alt,
        const unsigned int *d_tag,
        unsigned int *d_tag_alt,
        const Scalar4 *d_orientation,
        Scalar4 *d_orientation_alt,
        const Scalar4 *d_angmom,
        Scalar4 *d_angmom_alt,
        const Scalar3 *d_inertia,
        Scalar3 *d_inertia_alt,
        const Scalar *d_net_virial,
        Scalar *d_net_virial_alt,
        const Scalar4 *d_net_force,
        Scalar4 *d_net_force_alt,
        const Scalar4 *d_net_torque,
        Scalar4 *d_net_torque_alt,
        unsigned int *d_rtag);

#endif // __SFC_PACK_UPDATER_GPU_CUH__
