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
// Maintainer: ndtrung

/*! \file TwoStepNVERigidGPU.cuh
    \brief Declares GPU kernel code for NVE integration on the GPU. Used by TwoStepNVERigidGPU.
*/

#include "ParticleData.cuh"
#include "RigidData.cuh"

#ifndef __TWO_STEP_NVE_RIGID_GPU_CUH__
#define __TWO_STEP_NVE_RIGID_GPU_CUH__

//! Kernel driver for the first part of the NVE update called by TwoStepNVERigidGPU
cudaError_t gpu_nve_rigid_step_one(const gpu_pdata_arrays &pdata,
                             const gpu_rigid_data_arrays& rigid_data, 
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             float *d_net_virial,
                             const gpu_boxsize &box,
                             float deltaT);

//! Kernel driver for the second part of the NVE update called by TwoStepNVERigidGPU
cudaError_t gpu_nve_rigid_step_two(const gpu_pdata_arrays &pdata,
                             const gpu_rigid_data_arrays& rigid_data, 
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             float *d_net_virial,
                             const gpu_boxsize &box, 
                             float deltaT);

//! Kernel driver for the force and torque computes
cudaError_t gpu_rigid_force(const gpu_pdata_arrays &pdata,
                             const gpu_rigid_data_arrays& rigid_data, 
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             const gpu_boxsize &box, 
                             float deltaT);
                             
/*! Shared kernels for rigid body integrators
*/
                                                          
//! Kernel for the first step integration setting particle velocities called by TwoStepNVERigidGPU and TwoStepNVTRigidGPU
extern "C" __global__ void gpu_rigid_step_one_particle_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float *d_net_virial,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        gpu_boxsize box);

//! Kernel for the first step integration setting particle velocities called by TwoStepNVERigidGPU and TwoStepNVTRigidGPU for large bodies
extern "C" __global__ void gpu_rigid_step_one_particle_sliding_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float *d_net_virial,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        unsigned int nmax,
                                                        unsigned int block_size,
                                                        gpu_boxsize box);
                                                        
                                                 
//! Kernel for the second step integration setting particle velocities called by TwoStepNVERigidGPU and TwoStepNVTRigidGPU
extern "C" __global__ void gpu_rigid_step_two_particle_kernel(float4* pdata_vel,
                                                         float *d_net_virial,
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box);

//! Kernel for the second step integration setting particle velocities called by TwoStepNVERigidGPU and TwoStepNVTRigidGPU for large bodies
extern "C" __global__ void gpu_rigid_step_two_particle_sliding_kernel(float4* pdata_vel,
                                                         float *d_net_virial,
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         unsigned int block_size,
                                                         gpu_boxsize box);
                                                         

#endif //__TWO_STEP_NVE_RIGID_GPU_CUH__

