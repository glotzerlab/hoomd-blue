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

// Maintainer: askeys

#include "ParticleData.cuh"
#include "HOOMDMath.h"

#ifndef __FIRE_ENERGY_MINIMIZER_GPU_CUH__
#define __FIRE_ENERGY_MINIMIZER_GPU_CUH__

/*! \file FIREEnergyMinimizerGPU.cuh
    \brief Defines the interace to GPU kernal drivers used by FIREEnergyMinimizerGPU.
*/

//! Kernel driver for zeroing velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_zero_v(Scalar4 *d_vel,
                            unsigned int *d_group_members,
                            unsigned int group_size);

//! Kernel driver for summing the potential energy called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_compute_sum_pe(unsigned int *d_group_members,
                            unsigned int group_size,
                            Scalar4* d_net_force,
                            Scalar* d_sum_pe,
                            Scalar* d_partial_sum_pe,
                            unsigned int block_size,
                            unsigned int num_blocks);

//! Kernel driver for summing over P, vsq, and asq called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_compute_sum_all(const unsigned int N,
                            const Scalar4 *d_vel,
                            const Scalar3 *d_accel,
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            Scalar* d_sum_all,
                            Scalar* d_partial_sum_P,
                            Scalar* d_partial_sum_vsq,
                            Scalar* d_partial_sum_asq,
                            unsigned int block_size,
                            unsigned int num_blocks);

//! Kernel driver for updating the velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_update_v(Scalar4 *d_vel,
                            const Scalar3 *d_accel,
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            Scalar alpha,
                            Scalar vnorm,
                            Scalar invfnorm);

#endif //__FIRE_ENERGY_MINIMIZER_GPU_CUH__
