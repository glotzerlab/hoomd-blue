/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#ifndef __TWOSTEP_NPT_MTK_GPU_CUH__
#define __TWOSTEP_NPT_MTK_GPU_CUH__

#include <cuda_runtime.h>

#include "ParticleData.cuh"
#include "HOOMDMath.h"

/*! \file TwoStepNPTMTKGPU.cuh
    \brief Declares GPU kernel code for NPT integration on the GPU using the Martyna-Tobias-Klein (MTK) equations. Used by TwoStepNPTMTKGPU.
*/

//! Kernel driver for the the first step of the computation
cudaError_t gpu_npt_mtk_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar exp_thermo_fac,
                             Scalar *mat_exp_v,
                             Scalar *mat_exp_v_int,
                             Scalar *mat_exp_r,
                             Scalar *mat_exp_r_int,
                             Scalar deltaT,
                             bool rescale_all);

//! Kernel driver for wrapping particles back in the box (part of first step)
cudaError_t gpu_npt_mtk_wrap(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             const BoxDim& box);

//! Kernel driver for the the second step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_mtk_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             Scalar *mat_exp_v,
                             Scalar *mat_exp_v_int,
                             Scalar deltaT);

//! Kernel driver for reduction of temperature (part of second step)
cudaError_t gpu_npt_mtk_temperature(Scalar *d_temperature,
                                    Scalar4 *d_vel,
                                    Scalar *d_scratch,
                                    unsigned int num_blocks,
                                    unsigned int block_size,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    unsigned int ndof);

//! Kernel driver for rescaling of velocities (part of second step)
cudaError_t gpu_npt_mtk_thermostat(Scalar4 *d_vel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar xi,
                             Scalar deltaT);

//! Rescale all velocities
void gpu_npt_mtk_rescale(unsigned int N,
                       Scalar4 *d_postype,
                       Scalar mat_exp_r_xx,
                       Scalar mat_exp_r_xy,
                       Scalar mat_exp_r_xz,
                       Scalar mat_exp_r_yy,
                       Scalar mat_exp_r_yz,
                       Scalar mat_exp_r_zz);

#endif
