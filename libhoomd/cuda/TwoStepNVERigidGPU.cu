/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: ndtrung

#include "QuaternionMath.h"
#include "TwoStepNVERigidGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNVERigidGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by NVEUpdaterGPU.
*/

//! Flag for invalid particle index, identical to the sentinel value NO_INDEX in RigidData.h
#define INVALID_INDEX 0xffffffff 

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;
//! The texture for reading in the pdata mass array
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;

//! The texture for reading the rigid data body indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_body_indices_tex;
//! The texture for reading the rigid data body mass array
texture<float, 1, cudaReadModeElementType> rigid_data_body_mass_tex;
//! The texture for reading the rigid data moment of inertia array
texture<float4, 1, cudaReadModeElementType> rigid_data_moment_inertia_tex;
//! The texture for reading the rigid data com array
texture<float4, 1, cudaReadModeElementType> rigid_data_com_tex;
//! The texture for reading the rigid data vel array
texture<float4, 1, cudaReadModeElementType> rigid_data_vel_tex;
//! The texture for reading the rigid data angualr momentum array
texture<float4, 1, cudaReadModeElementType> rigid_data_angmom_tex;
//! The texture for reading the rigid data angular velocity array
texture<float4, 1, cudaReadModeElementType> rigid_data_angvel_tex;
//! The texture for reading the rigid data orientation array
texture<float4, 1, cudaReadModeElementType> rigid_data_orientation_tex;
//! The texture for reading the rigid data ex space array
texture<float4, 1, cudaReadModeElementType> rigid_data_exspace_tex;
//! The texture for reading the rigid data ey space array
texture<float4, 1, cudaReadModeElementType> rigid_data_eyspace_tex;
//! The texture for reading the rigid data ez space array
texture<float4, 1, cudaReadModeElementType> rigid_data_ezspace_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagex_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagey_tex;
//! The texture for reading in the rigid data body image array
texture<int, 1, cudaReadModeElementType> rigid_data_body_imagez_tex;
//! The texture for reading the rigid data particle position array
texture<float4, 1, cudaReadModeElementType> rigid_data_particle_pos_tex;
//! The texture for reading the rigid data particle indices array
texture<unsigned int, 1, cudaReadModeElementType> rigid_data_particle_indices_tex;
//! The texture for reading the rigid data force array
texture<float4, 1, cudaReadModeElementType> rigid_data_force_tex;
//! The texture for reading the rigid data torque array
texture<float4, 1, cudaReadModeElementType> rigid_data_torque_tex;
//! The texture for reading the rigid data particle old position array
texture<float4, 1, cudaReadModeElementType> rigid_data_particle_oldpos_tex;
//! The texture for reading the rigid data particle old velocity array
texture<float4, 1, cudaReadModeElementType> rigid_data_particle_oldvel_tex;

//! The texture for reading the rigid data conjugate qm array
texture<float4, 1, cudaReadModeElementType> rigid_data_conjqm_tex;

//! The texture for reading the net virial array
texture<float, 1, cudaReadModeElementType> net_virial_tex;

//! The texture for reading the net force array
texture<float4, 1, cudaReadModeElementType> net_force_tex;

//! The texture for reading the rigid virial array
texture<float, 1, cudaReadModeElementType> virial_tex;

/*! Takes the first half-step forward for rigid bodies in the velocity-verlet NVE integration
    \param rdata_com Body center of mass
    \param rdata_vel Body translational velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_orientation Quaternion
    \param rdata_ex_space x-axis unit vector
    \param rdata_ey_space y-axis unit vector
    \param rdata_ez_space z-axis unit vector
    \param rdata_body_imagex Body image in x-direction
    \param rdata_body_imagey Body image in y-direction
    \param rdata_body_imagez Body image in z-direction
    \param rdata_conjqm Conjugate quaternion momentum
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_step_one_body_kernel(float4* rdata_com, 
                                                        float4* rdata_vel, 
                                                        float4* rdata_angmom, 
                                                        float4* rdata_angvel,
                                                        float4* rdata_orientation, 
                                                        float4* rdata_ex_space, 
                                                        float4* rdata_ey_space, 
                                                        float4* rdata_ez_space, 
                                                        int* rdata_body_imagex, 
                                                        int* rdata_body_imagey, 
                                                        int* rdata_body_imagez,
                                                        float4* rdata_conjqm,  
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        gpu_boxsize box, 
                                                        float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    // do velocity verlet update
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    // r(t+deltaT) = r(t) + v(t+deltaT/2)*deltaT
    if (group_idx < n_group_bodies)
        {
        float body_mass;
        float4 moment_inertia, com, vel, angmom, orientation, ex_space, ey_space, ez_space, force, torque;
        int body_imagex, body_imagey, body_imagez;
        float dt_half = 0.5 * deltaT;
        
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        if (idx_body < n_bodies)
            {
            body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
            moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
            com = tex1Dfetch(rigid_data_com_tex, idx_body);
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
            orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
            body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
            body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
            body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
            force = tex1Dfetch(rigid_data_force_tex, idx_body);
            torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
         
            exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
               
            // update velocity
            float dtfm = dt_half / body_mass;
            
            float4 vel2;
            vel2.x = vel.x + dtfm * force.x;
            vel2.y = vel.y + dtfm * force.y;
            vel2.z = vel.z + dtfm * force.z;
            vel2.w = vel.w;
            
            // update position
            float4 pos2;
            pos2.x = com.x + vel2.x * deltaT;
            pos2.y = com.y + vel2.y * deltaT;
            pos2.z = com.z + vel2.z * deltaT;
            pos2.w = com.w;
            
            // read in body's image
            // time to fix the periodic boundary conditions
            float x_shift = rintf(pos2.x * box.Lxinv);
            pos2.x -= box.Lx * x_shift;
            body_imagex += (int)x_shift;
            
            float y_shift = rintf(pos2.y * box.Lyinv);
            pos2.y -= box.Ly * y_shift;
            body_imagey += (int)y_shift;
            
            float z_shift = rintf(pos2.z * box.Lzinv);
            pos2.z -= box.Lz * z_shift;
            body_imagez += (int)z_shift;
       
            // update the angular momentum and angular velocity
            float4 angmom2;
            angmom2.x = angmom.x + dt_half * torque.x;
            angmom2.y = angmom.y + dt_half * torque.y;
            angmom2.z = angmom.z + dt_half * torque.z;
            angmom2.w = 0.0;
            
            float4 angvel2;
            advanceQuaternion(angmom2, moment_inertia, angvel2, ex_space, ey_space, ez_space, deltaT, orientation);
       
            // write out the results (MEM_TRANSFER: ? bytes)
            rdata_com[idx_body] = pos2;
            rdata_vel[idx_body] = vel2;
            rdata_angmom[idx_body] = angmom2;
            rdata_angvel[idx_body] = angvel2;
            rdata_orientation[idx_body] = orientation;
            rdata_ex_space[idx_body] = ex_space;
            rdata_ey_space[idx_body] = ey_space;
            rdata_ez_space[idx_body] = ez_space;
            rdata_body_imagex[idx_body] = body_imagex;
            rdata_body_imagey[idx_body] = body_imagey;
            rdata_body_imagez[idx_body] = body_imagez;
            }
        }
    }

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param rdata_oldpos Particel old position
    \param rdata_oldvel Particel old velocity
    \param d_virial Virial contribution from the first part
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_nve_rigid_step_one_particle_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float4* rdata_oldpos,
                                                        float4* rdata_oldvel,
                                                        float *d_virial,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        gpu_boxsize box,
                                                        float deltaT)
    {
    
    unsigned int group_idx = blockIdx.x + local_beg;

    __shared__ float4 com, vel, angvel, ex_space, ey_space, ez_space;
    __shared__ int body_imagex, body_imagey, body_imagez;
    
    int idx_body = -1;    
    
    float dt_half = 0.5 * deltaT; 
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            com = tex1Dfetch(rigid_data_com_tex, idx_body);
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
            body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
            body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
            }
        }
        
    __syncthreads();
        
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        unsigned int idx_particle = idx_body * blockDim.x + threadIdx.x;
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);        
        // Since we use nmax for all rigid bodies, there might be some empty slot for particles in a rigid body
        // the particle index of these empty slots is set to be INVALID_INDEX.
        if (idx_particle_index != INVALID_INDEX)
            {
            float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            float4 particle_oldpos = tex1Dfetch(rigid_data_particle_oldpos_tex, idx_particle);
            float4 particle_oldvel = tex1Dfetch(rigid_data_particle_oldvel_tex, idx_particle);
            
            float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
            float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
            float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
            
            // compute ri with new orientation
            float4 ri;
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
            
            // x_particle = com + ri
            float4 ppos;
            ppos.x = com.x + ri.x;
            ppos.y = com.y + ri.y;
            ppos.z = com.z + ri.z;
            ppos.w = pos.w;
            
            // time to fix the periodic boundary conditions (FLOPS: 15)
            int4 image;
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
            
            // store unwrapped position
            Scalar4 unwrapped_pos;
            unwrapped_pos.x = ppos.x + box.Lx * image.x;
            unwrapped_pos.y = ppos.y + box.Ly * image.y;
            unwrapped_pos.z = ppos.z + box.Lz * image.z;
            
            // v_particle = vel + angvel x ri
            float4 pvel;
            pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
            pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
            pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
            pvel.w = 0.0f;
            
            float4 fc;
            fc.x = massone * (pvel.x - particle_oldvel.x) / dt_half - pforce.x;
            fc.y = massone * (pvel.y - particle_oldvel.y) / dt_half - pforce.y;
            fc.z = massone * (pvel.z - particle_oldvel.z) / dt_half - pforce.z; 
            
            float pvirial = 0.5f * (particle_oldpos.x * fc.x + particle_oldpos.y * fc.y + particle_oldpos.z * fc.z) / 3.0f;
            
            // write out the results (MEM_TRANSFER: ? bytes)
            pdata_pos[idx_particle_index] = ppos;
            pdata_vel[idx_particle_index] = pvel;
            pdata_image[idx_particle_index] = image;
            d_virial[idx_particle_index] = pvirial;
            rdata_oldpos[idx_particle] = unwrapped_pos;
            rdata_oldvel[idx_particle] = pvel;
            }
        }
    
    }

/*!
    \param pdata_pos Particle position
    \param pdata_vel Particle velocity
    \param pdata_image Particle image
    \param rdata_oldpos Particel old position
    \param rdata_oldvel Particel old velocity
    \param d_virial Virial contribution from the first part
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_nve_rigid_step_one_particle_sliding_kernel(float4* pdata_pos,
                                                        float4* pdata_vel,
                                                        int4* pdata_image,
                                                        float4* rdata_oldpos,
                                                        float4* rdata_oldvel,
                                                        float *d_virial,
                                                        unsigned int n_group_bodies,
                                                        unsigned int n_bodies, 
                                                        unsigned int local_beg,
                                                        unsigned int nmax,
                                                        unsigned int block_size,
                                                        gpu_boxsize box,
                                                        float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
        
    __shared__ float4 com, vel, angvel, ex_space, ey_space, ez_space;
    __shared__ int body_imagex, body_imagey, body_imagez;
    
    int idx_body = -1;
    
    float dt_half = 0.5 * deltaT; 
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
       
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            com = tex1Dfetch(rigid_data_com_tex, idx_body);
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            body_imagex = tex1Dfetch(rigid_data_body_imagex_tex, idx_body);
            body_imagey = tex1Dfetch(rigid_data_body_imagey_tex, idx_body);
            body_imagez = tex1Dfetch(rigid_data_body_imagez_tex, idx_body);
            }
        }
        
    __syncthreads();
        
    unsigned int n_windows = nmax / block_size + 1;
    for (unsigned int start = 0; start < n_windows; start++)
        {
        if (idx_body >= 0 && idx_body < n_bodies)
            {
            int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
            if (idx_particle < nmax * n_bodies && start * block_size + threadIdx.x < nmax)
                {
                unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);  
                if (idx_particle_index != INVALID_INDEX)
                    {
                    float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    float4 particle_oldpos = tex1Dfetch(rigid_data_particle_oldpos_tex, idx_particle);
                    float4 particle_oldvel = tex1Dfetch(rigid_data_particle_oldvel_tex, idx_particle);
                    
                    float4 pos = tex1Dfetch(pdata_pos_tex, idx_particle_index);
                    float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
                    float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
                    
                    // compute ri with new orientation
                    float4 ri;
                    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
                    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
                    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
                    
                    // x_particle = com + ri
                    float4 ppos;
                    ppos.x = com.x + ri.x;
                    ppos.y = com.y + ri.y;
                    ppos.z = com.z + ri.z;
                    ppos.w = pos.w;
                    
                    // time to fix the periodic boundary conditions (FLOPS: 15)
                    int4 image;
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
                    
                    // store unwrapped position
                    Scalar4 unwrapped_pos;
                    unwrapped_pos.x = ppos.x + box.Lx * image.x;
                    unwrapped_pos.y = ppos.y + box.Ly * image.y;
                    unwrapped_pos.z = ppos.z + box.Lz * image.z;
                    
                    // v_particle = vel + angvel x ri
                    float4 pvel;
                    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
                    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
                    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
                    pvel.w = 0.0f;
                    
                    float4 fc;
                    fc.x = massone * (pvel.x - particle_oldvel.x) / dt_half - pforce.x;
                    fc.y = massone * (pvel.y - particle_oldvel.y) / dt_half - pforce.y;
                    fc.z = massone * (pvel.z - particle_oldvel.z) / dt_half - pforce.z; 
                    
                    float pvirial = 0.5f * (particle_oldpos.x * fc.x + particle_oldpos.y * fc.y + particle_oldpos.z * fc.z) / 3.0f;
                    
                    // write out the results (MEM_TRANSFER: ? bytes)
                    pdata_pos[idx_particle_index] = ppos;
                    pdata_vel[idx_particle_index] = pvel;
                    pdata_image[idx_particle_index] = image;
                    d_virial[idx_particle_index] = pvirial;
                    rdata_oldpos[idx_particle] = unwrapped_pos;
                    rdata_oldvel[idx_particle] = pvel;
                    }
                }
            }
        }
    }

// Takes the first 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_one(const gpu_pdata_arrays& pdata, 
                                   const gpu_rigid_data_arrays& rigid_data, 
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   const gpu_boxsize &box, 
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for rigid bodies:
    // body mass, com, vel, angmom, angvel, orientation, ex_space, ey_space, ez_space, body images, particle pos, particle indices, force and torque
    
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rigid_data.body_indices, sizeof(unsigned int) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_orientation_tex, rigid_data.orientation, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagex_tex, rigid_data.body_imagex, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagey_tex, rigid_data.body_imagey, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagez_tex, rigid_data.body_imagez, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_force_tex, rigid_data.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_torque_tex, rigid_data.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_particle_oldpos_tex, rigid_data.particle_oldpos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_particle_oldvel_tex, rigid_data.particle_oldvel, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    // setup the grid to run the kernel for rigid bodies
    int block_size = 64;
    int n_blocks = n_group_bodies / block_size + 1;
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);
    
    gpu_nve_rigid_step_one_body_kernel<<< body_grid, body_threads >>>(rigid_data.com, 
                                                           rigid_data.vel, 
                                                           rigid_data.angmom, 
                                                           rigid_data.angvel,
                                                           rigid_data.orientation, 
                                                           rigid_data.ex_space, 
                                                           rigid_data.ey_space, 
                                                           rigid_data.ez_space, 
                                                           rigid_data.body_imagex, 
                                                           rigid_data.body_imagey, 
                                                           rigid_data.body_imagez,
                                                           rigid_data.conjqm,
                                                           n_group_bodies, 
                                                           n_bodies, 
                                                           local_beg,
                                                           box, 
                                                           deltaT);
    
    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_body_imagex_tex, rigid_data.body_imagex, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagey_tex, rigid_data.body_imagey, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_imagez_tex, rigid_data.body_imagez, sizeof(int) * n_bodies);
    if (error != cudaSuccess)
        return error;

    // bind the textures for particles: pos, vel and image of ALL particles (remember pos.w is the partice type needed to for new positions)
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error; 

    if (nmax <= 32)
        {
        block_size = nmax; // maximum number of particles in a rigid body: each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_nve_rigid_step_one_particle_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     rigid_data.particle_oldpos,
                                                                     rigid_data.particle_oldvel,
                                                                     rigid_data.virial,
                                                                     n_group_bodies,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     box, 
                                                                     deltaT);
        }
    else
        {
        block_size = 32; 	// chosen to be divisible by nmax
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);
        
        gpu_nve_rigid_step_one_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.pos, 
                                                                     pdata.vel, 
                                                                     pdata.image,
                                                                     rigid_data.particle_oldpos,
                                                                     rigid_data.particle_oldvel,
                                                                     rigid_data.virial,
                                                                     n_group_bodies,
                                                                     n_bodies, 
                                                                     local_beg,
                                                                     nmax,
                                                                     block_size,
                                                                     box, 
                                                                     deltaT);
        }
    
    return cudaSuccess;
    }

    
#pragma mark RIGID_FORCE_KERNEL

//! Shared memory for body force and torque reduction, required allocation when the kernel is called
extern __shared__ float4 sum[];

//! Calculates the body forces and torques by summing the constituent particle forces
/*! \param rdata_force Body forces
    \param rdata_torque Body torques
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_force_kernel(float4* rdata_force, 
                                                 float4* rdata_torque, 
                                                 unsigned int n_group_bodies,
                                                 unsigned int n_bodies, 
                                                 unsigned int local_beg,
                                                 unsigned int nmax,
                                                 gpu_boxsize box)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 ex_space, ey_space, ez_space;
    float4 *body_force = sum;
    float4 *body_torque = &sum[blockDim.x];
        
    body_force[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    body_torque[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 fi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 torquei = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
    int idx_body = -1;
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            // calculate body force and torques
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
        
    __syncthreads();
    
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        unsigned int idx_particle = idx_body * blockDim.x + threadIdx.x; 
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);    
        if (idx_body < n_bodies && idx_particle_index != INVALID_INDEX)
            {
            // calculate body force and torques
            float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            fi = tex1Dfetch(net_force_tex, idx_particle_index);
            
            body_force[threadIdx.x].x = fi.x;
            body_force[threadIdx.x].y = fi.y;
            body_force[threadIdx.x].z = fi.z;
            body_force[threadIdx.x].w = fi.w;
                    
            // This might require more calculations but more stable 
            // particularly when rigid bodies are bigger than half the box
            float4 ri;
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y 
                    + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y 
                    + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y 
                    + ez_space.z * particle_pos.z;
            
            torquei.x = ri.y * fi.z - ri.z * fi.y;
            torquei.y = ri.z * fi.x - ri.x * fi.z;
            torquei.z = ri.x * fi.y - ri.y * fi.x;
            torquei.w = 0.0;
            
            body_torque[threadIdx.x].x = torquei.x;
            body_torque[threadIdx.x].y = torquei.y;
            body_torque[threadIdx.x].z = torquei.z;
            body_torque[threadIdx.x].w = torquei.w;
            }
        }
            
    __syncthreads();
    
    unsigned int offset = blockDim.x >> 1;
    while (offset > 0)
        {
        if (threadIdx.x < offset)
            {
            body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
            body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
            body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
            body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;
            
            body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
            body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
            body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
            body_torque[threadIdx.x].w += body_torque[threadIdx.x + offset].w;
            }
            
        offset >>= 1;
        
        __syncthreads();
        }

    if (idx_body >= 0 && idx_body < n_bodies)
        {
        // Every thread now has its own copy of body force and torque
        float4 force2 = body_force[0];
        float4 torque2 = body_torque[0];
        
        if (threadIdx.x == 0)
            {
            rdata_force[idx_body] = force2;
            rdata_torque[idx_body] = torque2;
            }
        }   
    }

//! Calculates the body forces and torques by summing the constituent particle forces using a fixed sliding window size
/*! \param rdata_force Body forces
    \param rdata_torque Body torques
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param window_size Window size for reduction
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_rigid_force_sliding_kernel(float4* rdata_force, 
                                                 float4* rdata_torque,
                                                 unsigned int n_group_bodies, 
                                                 unsigned int n_bodies, 
                                                 unsigned int local_beg,
                                                 unsigned int nmax,
                                                 unsigned int window_size,
                                                 gpu_boxsize box)
    {
    int group_idx = blockIdx.x + local_beg;
    
    float4 *body_force = sum;
    float4 *body_torque = &sum[window_size];
        
    __shared__ float4 ex_space, ey_space, ez_space;
    float4 force2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 
    float4 torque2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 
    
    int idx_body = -1;
            
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
                  
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
                
    __syncthreads();
        
    unsigned int n_windows = nmax / window_size;
        
    // slide the window throughout the block
    for (unsigned int start = 0; start < n_windows; start++)
        {
        body_force[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        body_torque[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 fi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 torquei = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (idx_body >= 0 && idx_body < n_bodies)
            { 
            int idx_particle = idx_body * nmax + start * window_size + threadIdx.x;
            unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
                     
            if (idx_particle_index != INVALID_INDEX)
                {
                // calculate body force and torques
                float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                fi = tex1Dfetch(net_force_tex, idx_particle_index);
                                
                body_force[threadIdx.x].x = fi.x;
                body_force[threadIdx.x].y = fi.y;
                body_force[threadIdx.x].z = fi.z;
                body_force[threadIdx.x].w = fi.w;
                
                // This might require more calculations but more stable
                // particularly when rigid bodies are bigger than half the box
                float4 ri;
                ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y 
                        + ez_space.x * particle_pos.z;
                ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y 
                        + ez_space.y * particle_pos.z;
                ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y 
                        + ez_space.z * particle_pos.z;

                torquei.x = ri.y * fi.z - ri.z * fi.y;
                torquei.y = ri.z * fi.x - ri.x * fi.z;
                torquei.z = ri.x * fi.y - ri.y * fi.x;
                torquei.w = 0.0;
                
                body_torque[threadIdx.x].x = torquei.x;
                body_torque[threadIdx.x].y = torquei.y;
                body_torque[threadIdx.x].z = torquei.z;
                body_torque[threadIdx.x].w = torquei.w;
                }
            }
            
        __syncthreads();

        // reduction within the current window
        unsigned int offset = window_size >> 1;
        while (offset > 0)
            {
            if (threadIdx.x < offset)
                {
                body_force[threadIdx.x].x += body_force[threadIdx.x + offset].x;
                body_force[threadIdx.x].y += body_force[threadIdx.x + offset].y;
                body_force[threadIdx.x].z += body_force[threadIdx.x + offset].z;
                body_force[threadIdx.x].w += body_force[threadIdx.x + offset].w;
                
                body_torque[threadIdx.x].x += body_torque[threadIdx.x + offset].x;
                body_torque[threadIdx.x].y += body_torque[threadIdx.x + offset].y;
                body_torque[threadIdx.x].z += body_torque[threadIdx.x + offset].z;
                body_torque[threadIdx.x].w += body_torque[threadIdx.x + offset].w;
                }
                
            offset >>= 1;
            
            __syncthreads();
            }
        
        // accumulate the body force into the thread-local variables
        force2.x += body_force[0].x;
        force2.y += body_force[0].y;
        force2.z += body_force[0].z;
        force2.w += body_force[0].w;
        
        torque2.x += body_torque[0].x;
        torque2.y += body_torque[0].y;
        torque2.z += body_torque[0].z;
        torque2.w += body_torque[0].w;
        
        __syncthreads();
        }
        
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        if (threadIdx.x == 0)
            {
            rdata_force[idx_body] = force2;
            rdata_torque[idx_body] = torque2;
            }
        }
    
			
    }

/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_rigid_force(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size, 
                                   float4 *d_net_force,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for ALL rigid bodies
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rigid_data.body_indices, sizeof(unsigned int) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
            
    error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;    
    
    // bind the textures for particle forces
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;   
        
    // run the kernel: the shared memory size is used for dynamic memory allocation of extern __shared__ sum
    // 32 is some threshold for really big rigid bodies- this should be small to reduce overhead in required shared memory
    if (nmax <= 32)
		{
		unsigned int block_size = nmax; // each thread in a block takes care of a particle in a rigid body
		dim3 force_grid(n_bodies, 1, 1);
		dim3 force_threads(block_size, 1, 1); 
    
        gpu_rigid_force_kernel<<< force_grid, force_threads, 2 * nmax * sizeof(float4) >>>(rigid_data.force, 
                                                                                             rigid_data.torque,
                                                                                             n_group_bodies, 
                                                                                             n_bodies, 
                                                                                             local_beg,
                                                                                             nmax,
                                                                                             box);
		}
	else	// large rigid bodies
		{
		unsigned int window_size = 16;			// some fixed value divisble by nmax
		unsigned int block_size = window_size;
		dim3 force_grid(n_bodies, 1, 1);
		dim3 force_threads(block_size, 1, 1); 
    
	    gpu_rigid_force_sliding_kernel<<< force_grid, force_threads, 2 * window_size * sizeof(float4) >>>(rigid_data.force, 
                                                                                             rigid_data.torque,
                                                                                             n_group_bodies, 
                                                                                             n_bodies, 
                                                                                             local_beg,
                                                                                             nmax,
                                                                                             window_size,
                                                                                             box);       
		}
	
    return cudaSuccess;
    }


#pragma mark RIGID_STEP_TWO_KERNEL

/*! Takes the second half-step forward for rigid bodies in the velocity-verlet NVE integration
    \param rdata_vel Body translational velocity
    \param rdata_angmom Angular momentum
    \param rdata_angvel Angular velocity
    \param rdata_conjqm Conjugate quaternion momentum
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param deltaT Timestep 
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ void gpu_nve_rigid_step_two_body_kernel(float4* rdata_vel, 
                                                         float4* rdata_angmom, 
                                                         float4* rdata_angvel,
                                                         float4* rdata_conjqm, 
                                                         unsigned int n_group_bodies,
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box, 
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x * blockDim.x + threadIdx.x + local_beg;
    
    if (group_idx < n_group_bodies)
        {
        float body_mass;
        float4 moment_inertia, vel, angmom, orientation, ex_space, ey_space, ez_space, force, torque;
        float dt_half = 0.5 * deltaT;
        
        unsigned int idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
            
        // Update body velocity and angmom
        if (idx_body < n_bodies)
            {        
            // update the velocity
            body_mass = tex1Dfetch(rigid_data_body_mass_tex, idx_body);
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angmom = tex1Dfetch(rigid_data_angmom_tex, idx_body);
            force = tex1Dfetch(rigid_data_force_tex, idx_body);
            torque = tex1Dfetch(rigid_data_torque_tex, idx_body);
            moment_inertia = tex1Dfetch(rigid_data_moment_inertia_tex, idx_body);
            orientation = tex1Dfetch(rigid_data_orientation_tex, idx_body);
            
            exyzFromQuaternion(orientation, ex_space, ey_space, ez_space);
              
            float dtfm = dt_half / body_mass;
            float4 vel2;
            vel2.x = vel.x + dtfm * force.x;
            vel2.y = vel.y + dtfm * force.y;
            vel2.z = vel.z + dtfm * force.z;
            vel2.w = 0.0;

            // update angular momentum
            float4 angmom2;
            angmom2.x = angmom.x + dt_half * torque.x;
            angmom2.y = angmom.y + dt_half * torque.y;
            angmom2.z = angmom.z + dt_half * torque.z;
            angmom2.w = 0.0;
            
            // update angular velocity        
            float4 angvel2;
            computeAngularVelocity(angmom2, moment_inertia, ex_space, ey_space, ez_space, angvel2);
            
            // write out results
            rdata_vel[idx_body] = vel2;
            rdata_angmom[idx_body] = angmom2;
            rdata_angvel[idx_body] = angvel2;
            }
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param rdata_oldvel Particle velocity from the previous step
    \param d_net_virial Particle virial
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_nve_rigid_step_two_particle_kernel(float4* pdata_vel,
                                                         float4* rdata_oldvel,
                                                         float *d_net_virial,
                                                         unsigned int n_group_bodies,
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         gpu_boxsize box,
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 vel, angvel, ex_space, ey_space, ez_space;
    
    float dt_half = 0.5 * deltaT;
    
    int idx_body = -1;    

    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
        
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
        
    __syncthreads();
    
    if (idx_body >= 0 && idx_body < n_bodies)
        {
        unsigned int idx_particle = idx_body * blockDim.x + threadIdx.x;
        unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
        if (idx_particle_index != INVALID_INDEX)
            {
            float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
            float4 particle_oldpos = tex1Dfetch(rigid_data_particle_oldpos_tex, idx_particle);
            float4 particle_oldvel = tex1Dfetch(rigid_data_particle_oldvel_tex, idx_particle);
            
            float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
            float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
            float net_virial = tex1Dfetch(net_virial_tex, idx_particle_index);
            float virial = tex1Dfetch(virial_tex, idx_particle_index);
            
            float4 ri;
            ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
            ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
            ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
            
            // v_particle = v_com + angvel x xr
            float4 pvel;
            pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
            pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
            pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
            pvel.w = 0.0;
            
            float4 fc;
            fc.x = massone * (pvel.x - particle_oldvel.x) / dt_half - pforce.x;
            fc.y = massone * (pvel.y - particle_oldvel.y) / dt_half - pforce.y;
            fc.z = massone * (pvel.z - particle_oldvel.z) / dt_half - pforce.z; 
        
            float pvirial = 0.5f * (particle_oldpos.x * fc.x + particle_oldpos.y * fc.y + particle_oldpos.z * fc.z) / 3.0f;
            
            // accumulate the virial contribution from the first part into the net particle virial
            pvirial += virial;
            pvirial += net_virial;
            
            // write out the results
            pdata_vel[idx_particle_index] = pvel;
            d_net_virial[idx_particle_index] = pvirial;
            rdata_oldvel[idx_particle] = pvel;
            }
        }
    }

/*!
    \param pdata_vel Particle velocity
    \param rdata_oldvel Particle velocity from the previous step
    \param d_net_virial Particle virial
    \param n_group_bodies Number of rigid bodies in my group
    \param n_bodies Total number of rigid bodies
    \param local_beg Starting body index in this card
    \param nmax Maximum number of particles in a rigid body
    \param block_size Block size
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Time step
*/
extern "C" __global__ void gpu_nve_rigid_step_two_particle_sliding_kernel(float4* pdata_vel,
                                                         float4* rdata_oldvel,
                                                         float *d_net_virial,
                                                         unsigned int n_group_bodies,   
                                                         unsigned int n_bodies, 
                                                         unsigned int local_beg,
                                                         unsigned int nmax,
                                                         unsigned int block_size,
                                                         gpu_boxsize box,
                                                         float deltaT)
    {
    unsigned int group_idx = blockIdx.x + local_beg;
    
    __shared__ float4 vel, angvel, ex_space, ey_space, ez_space;

    float dt_half = 0.5 * deltaT;
    
    int idx_body = -1;
    
    if (group_idx < n_group_bodies)
        {
        idx_body = tex1Dfetch(rigid_data_body_indices_tex, group_idx);
    
        if (idx_body < n_bodies && threadIdx.x == 0)
            {
            vel = tex1Dfetch(rigid_data_vel_tex, idx_body);
            angvel = tex1Dfetch(rigid_data_angvel_tex, idx_body);
            ex_space = tex1Dfetch(rigid_data_exspace_tex, idx_body);
            ey_space = tex1Dfetch(rigid_data_eyspace_tex, idx_body);
            ez_space = tex1Dfetch(rigid_data_ezspace_tex, idx_body);
            }
        }
        
    __syncthreads();
    
    unsigned int n_windows = nmax / block_size + 1;
    
    for (unsigned int start = 0; start < n_windows; start++)
        {
        if (idx_body >= 0 && idx_body < n_bodies)
            {
            int idx_particle = idx_body * nmax + start * block_size + threadIdx.x;
            if (idx_particle < nmax * n_bodies && start * block_size + threadIdx.x < nmax)
                {
                unsigned int idx_particle_index = tex1Dfetch(rigid_data_particle_indices_tex, idx_particle);
            
                if (idx_particle_index != INVALID_INDEX)
                    {
                    float4 particle_pos = tex1Dfetch(rigid_data_particle_pos_tex, idx_particle);
                    float4 particle_oldpos = tex1Dfetch(rigid_data_particle_oldpos_tex, idx_particle);
                    float4 particle_oldvel = tex1Dfetch(rigid_data_particle_oldvel_tex, idx_particle);
            
                    float massone = tex1Dfetch(pdata_mass_tex, idx_particle_index);
                    float4 pforce = tex1Dfetch(net_force_tex, idx_particle_index);
                    float net_virial = tex1Dfetch(net_virial_tex, idx_particle_index);
                    float virial = tex1Dfetch(virial_tex, idx_particle_index);
                    
                    float4 ri;
                    ri.x = ex_space.x * particle_pos.x + ey_space.x * particle_pos.y + ez_space.x * particle_pos.z;
                    ri.y = ex_space.y * particle_pos.x + ey_space.y * particle_pos.y + ez_space.y * particle_pos.z;
                    ri.z = ex_space.z * particle_pos.x + ey_space.z * particle_pos.y + ez_space.z * particle_pos.z;
                    
                    // v_particle = v_com + angvel x xr
                    float4 pvel;
                    pvel.x = vel.x + angvel.y * ri.z - angvel.z * ri.y;
                    pvel.y = vel.y + angvel.z * ri.x - angvel.x * ri.z;
                    pvel.z = vel.z + angvel.x * ri.y - angvel.y * ri.x;
                    pvel.w = 0.0;
                    
                    float4 fc;
                    fc.x = massone * (pvel.x - particle_oldvel.x) / dt_half - pforce.x;
                    fc.y = massone * (pvel.y - particle_oldvel.y) / dt_half - pforce.y;
                    fc.z = massone * (pvel.z - particle_oldvel.z) / dt_half - pforce.z; 
                    
                    float pvirial = 0.5f * (particle_oldpos.x * fc.x + particle_oldpos.y * fc.y + particle_oldpos.z * fc.z) / 3.0f;
                
                    // accumulate the virial contribution from the first part into the net particle virial
                    pvirial += virial;
                    pvirial += net_virial;
                    
                    // write out the results
                    pdata_vel[idx_particle_index] = pvel;
                    d_net_virial[idx_particle_index] = pvirial;
                    rdata_oldvel[idx_particle] = pvel;
                    }
                }
            }
        }

    }


// Take the second 1/2 step forward in the NVE integration step
/*! \param pdata Particle data to step forward 1/2 step
    \param rigid_data Rigid body data to step forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Particle net forces
    \param d_net_virial Particle net virial
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nve_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                   const gpu_rigid_data_arrays& rigid_data,
                                   unsigned int *d_group_members,
                                   unsigned int group_size,
                                   float4 *d_net_force,
                                   float *d_net_virial,
                                   const gpu_boxsize &box,
                                   float deltaT)
    {
    unsigned int n_bodies = rigid_data.n_bodies;
    unsigned int n_group_bodies = rigid_data.n_group_bodies;
    unsigned int local_beg = rigid_data.local_beg;
    unsigned int nmax = rigid_data.nmax;
    
    // bind the textures for ALL rigid bodies
    cudaError_t error = cudaBindTexture(0, rigid_data_body_indices_tex, rigid_data.body_indices, sizeof(unsigned int) * n_group_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_body_mass_tex, rigid_data.body_mass, sizeof(float) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_moment_inertia_tex, rigid_data.moment_inertia, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_com_tex, rigid_data.com, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angmom_tex, rigid_data.angmom, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_orientation_tex, rigid_data.orientation, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_exspace_tex, rigid_data.ex_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_eyspace_tex, rigid_data.ey_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_ezspace_tex, rigid_data.ez_space, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_pos_tex, rigid_data.particle_pos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_particle_indices_tex, rigid_data.particle_indices, sizeof(unsigned int) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_force_tex, rigid_data.force, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_torque_tex, rigid_data.torque, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_particle_oldpos_tex, rigid_data.particle_oldpos, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, rigid_data_particle_oldvel_tex, rigid_data.particle_oldvel, sizeof(float4) * n_bodies * nmax);
    if (error != cudaSuccess)
        return error;
        
    // bind the textures for particles
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;        
   
     unsigned int block_size = 64;
    unsigned int n_blocks = n_group_bodies / block_size + 1;                                
    dim3 body_grid(n_blocks, 1, 1);
    dim3 body_threads(block_size, 1, 1);                                                 
    gpu_nve_rigid_step_two_body_kernel<<< body_grid, body_threads >>>(rigid_data.vel, 
                                                                      rigid_data.angmom, 
                                                                      rigid_data.angvel,
                                                                      rigid_data.conjqm,
                                                                      n_group_bodies,
                                                                      n_bodies, 
                                                                      local_beg,
                                                                      nmax, 
                                                                      box, 
                                                                      deltaT);
    
    // get the body information after the above update
    error = cudaBindTexture(0, rigid_data_vel_tex, rigid_data.vel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, rigid_data_angvel_tex, rigid_data.angvel, sizeof(float4) * n_bodies);
    if (error != cudaSuccess)
        return error;
    
    // get the particle information
    error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error; 
        
    error = cudaBindTexture(0, net_virial_tex, d_net_virial, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, virial_tex, rigid_data.virial, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    if (nmax <= 32)
        {                                                                                                                                    
        block_size = nmax; // each thread in a block takes care of a particle in a rigid body
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_nve_rigid_step_two_particle_kernel<<< particle_grid, particle_threads >>>(pdata.vel,
                                                        rigid_data.particle_oldvel,
                                                        d_net_virial,
                                                        n_group_bodies,
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax, 
                                                        box,
                                                        deltaT);
        }
    else
        {
        block_size = 32; 
        dim3 particle_grid(n_group_bodies, 1, 1);
        dim3 particle_threads(block_size, 1, 1);                                                
        gpu_nve_rigid_step_two_particle_sliding_kernel<<< particle_grid, particle_threads >>>(pdata.vel,
                                                        rigid_data.particle_oldvel,
                                                        d_net_virial, 
                                                        n_group_bodies,
                                                        n_bodies, 
                                                        local_beg,
                                                        nmax,
                                                        block_size, 
                                                        box,
                                                        deltaT);
        }            
           
    return cudaSuccess;
    }
