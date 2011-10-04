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

// Maintainer: askeys

#include "ParticleData.cuh"

#ifndef __FIRE_ENERGY_MINIMIZER_GPU_CUH__
#define __FIRE_ENERGY_MINIMIZER_GPU_CUH__

/*! \file FIREEnergyMinimizerGPU.cuh
    \brief Defines the interace to GPU kernal drivers used by FIREEnergyMinimizerGPU.
*/

//! Kernel driver for zeroing velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_zero_v(gpu_pdata_arrays pdata,
                            unsigned int *d_group_members,
                            unsigned int group_size); 

//! Kernel driver for summing the potential energy called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_compute_sum_pe(const gpu_pdata_arrays& pdata, 
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            float4* d_net_force, 
                            float* d_sum_pe, 
                            float* d_partial_sum_pe, 
                            unsigned int block_size, 
                            unsigned int num_blocks);

//! Kernel driver for summing over P, vsq, and asq called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_compute_sum_all(const gpu_pdata_arrays& pdata, 
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            float* d_sum_all, 
                            float* d_partial_sum_P, 
                            float* d_partial_sum_vsq, 
                            float* d_partial_sum_asq, 
                            unsigned int block_size, 
                            unsigned int num_blocks);
                            
//! Kernel driver for updating the velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_update_v(gpu_pdata_arrays pdata, 
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            float alpha, 
                            float vnorm, 
                            float invfnorm);

#endif //__FIRE_ENERGY_MINIMIZER_GPU_CUH__

