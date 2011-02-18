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
                             float4 *d_net_torque,
                             const gpu_boxsize &box, 
                             float deltaT);

#ifdef NVCC
//! Kernel shared by other NVT and NPT rigid integrators
/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param d_pgroup_idx Particle index
    \param rdata_oldpos Particel old position
    \param rdata_oldvel Particel old velocity
    \param d_virial Virial contribution from the first part
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
template<bool set_x>
__global__ void gpu_rigid_setxv_kernel(float4* pdata_pos,
                                       float4* pdata_vel,
                                       float4* pdata_orientation,
                                       int4* pdata_image,
                                       unsigned int *d_pgroup_idx,
                                       unsigned int n_pgroup,
                                       unsigned int *d_particle_offset,
                                       unsigned int *d_particle_body,
                                       unsigned int *d_rigid_group,
                                       float4* d_rigid_orientation,
                                       float4* d_rigid_com,
                                       float4* d_rigid_vel,
                                       float4* d_rigid_angvel,
                                       int* d_rigid_imagex,
                                       int* d_rigid_imagey,
                                       int* d_rigid_imagez,
                                       unsigned int* d_rigid_particle_idx,
                                       float4* d_rigid_particle_dis,
                                       float4* d_rigid_particle_orientation,
                                       unsigned int n_group_bodies,
                                       unsigned int n_particles,
                                       unsigned int nmax,
                                       gpu_boxsize box)
    {
    float4 com, vel, angvel, ex_space, ey_space, ez_space;
    int body_imagex=0, body_imagey=0, body_imagez=0;

    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= n_pgroup)
        return;
    
    unsigned int pidx = d_pgroup_idx[group_idx];
    
    unsigned int idx_body = d_particle_body[pidx];
    unsigned int particle_offset = d_particle_offset[pidx];
    float4 orientation = d_rigid_orientation[idx_body];
    com = d_rigid_com[idx_body];
    vel = d_rigid_vel[idx_body];
    angvel = d_rigid_angvel[idx_body];
    if (set_x)
        {
        body_imagex = d_rigid_imagex[idx_body];
        body_imagey = d_rigid_imagey[idx_body];
        body_imagez = d_rigid_imagez[idx_body];
        }
    
    exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
    
    int localidx = idx_body * nmax + particle_offset;
    float4 particle_pos = d_rigid_particle_dis[localidx];
    
    // compute ri with new orientation
    float4 ri;
    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
    
    float4 ppos;
    int4 image;
    if (set_x)
        {
        // x_particle = com + ri
        ppos.x = com.x + ri.x;
        ppos.y = com.y + ri.y;
        ppos.z = com.z + ri.z;
        ppos.w = pdata_pos[pidx].w;
        
        // time to fix the periodic boundary conditions
        float x_shift = rintf(ppos.x * box.Lxinv);
        ppos.x -= box.Lx * x_shift;
        image.x = body_imagex;
        image.x += (int)x_shift;
        
        float y_shift = rintf(ppos.y * box.Lyinv);
        ppos.y -= box.Ly * y_shift;
        image.y = body_imagey;
        image.y += (int)y_shift;
        
        float z_shift = rintf(ppos.z * box.Lzinv);
        ppos.z -= box.Lz * z_shift;
        image.z = body_imagez;
        image.z += (int)z_shift;

        //update particle orientation
        quatquat(d_rigid_orientation[idx_body],
                 d_rigid_particle_orientation[localidx],
                 p_data_orientation[pidx]);
        }
    
    // v_particle = vel + angvel x ri
    float4 pvel;
    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
    pvel.w = 0.0f;
    
    // write out the results
    if (set_x)
        {
        pdata_pos[pidx] = ppos;
        pdata_image[pidx] = image;
        pdata
        }
    pdata_vel[pidx] = pvel;
    }
#endif

#endif //__TWO_STEP_NVE_RIGID_GPU_CUH__

