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

#include "TwoStepNPTGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPTGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTGPU.
*/

//! Texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! Texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! Texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading particle mass
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;
//! The texture for reading particle image
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;

//! Shared data used by NPT kernels for sum reductions
extern __shared__ float npt_sdata[];

/*! \param pdata Particle data arrays to integrate forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param partial_scale Set to true to only scale those particles in the group
    \param exp_v_fac exp_v_fac = \f$\exp(-\frac 1 4 (\eta+\xi)*\delta T)\f$ is the scaling factor for
velocity update and is a result of coupling to the thermo/barostat
    \param exp_r_fac exp_r_fac = \f$\exp(\frac 1 2 \eta\delta T)\f$ is the scaling factor for
position update and is a result of coupling to the thermo/barostat
    \param deltaT Time to advance (for one full step)
*/
extern "C" __global__ 
void gpu_npt_step_one_kernel(gpu_pdata_arrays pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             bool partial_scale,
                             float exp_v_fac,
                             float exp_r_fac,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // loop invariant quantities
    float exp_r_fac_inv = 1.0f / exp_r_fac;
    
    // propagate velocity from t to t+1/2*deltaT and position from t to t+deltaT
    // according to the Nose-Hoover barostat
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        
        // fetch particle position
        float4 pos = tex1Dfetch(pdata_pos_tex, idx);
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        // fetch particle velocity and acceleration
        float4 vel = tex1Dfetch(pdata_vel_tex, idx);
        float4 accel = tex1Dfetch(pdata_accel_tex, idx);
        
        // propagate velocity by half a time step and position by the full time step
        // according to the Nose-Hoover barostat
        vel.x = vel.x*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.x;
        px = px + vel.x*exp_r_fac_inv*deltaT;
        
        vel.y = vel.y*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.y;
        py = py + vel.y*exp_r_fac_inv*deltaT;
        
        vel.z = vel.z*exp_v_fac*exp_v_fac + (1.0f/2.0f) * deltaT*exp_v_fac*accel.z;
        pz = pz + vel.z*exp_r_fac_inv*deltaT;
        
        if (partial_scale)
            {
            px *= exp_r_fac*exp_r_fac;
            py *= exp_r_fac*exp_r_fac;
            pz *= exp_r_fac*exp_r_fac;
            }
        
        float4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;
        
        // write out the results
        pdata.pos[idx] = pos2;
        pdata.vel[idx] = vel;
        }
    }

/*! \param pdata Particle Data to operate on
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param partial_scale Set to true to only scale those particles in the group
    \param Xi theromstat variable in Nose-Hoover barostat
    \param Eta barostat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_step_one_kernel(). See it for more details.
*/
cudaError_t gpu_npt_step_one(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             bool partial_scale,
                             float Xi,
                             float Eta,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_accel_tex, pdata.accel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // precalculate scaling factors for baro/thermostat
    float exp_v_fac = exp(-1.0f/4.0f*(Eta+Xi)*deltaT);  // velocity scaling
    float exp_r_fac = exp(1.0f/2.0f*Eta*deltaT);        // position scaling
    
    // run the kernel
    gpu_npt_step_one_kernel<<< grid, threads >>>(pdata,
                                                 d_group_members,
                                                 group_size,
                                                 partial_scale,
                                                 exp_v_fac,
                                                 exp_r_fac,
                                                 deltaT);
    
    return cudaSuccess;
    }
    
/*! \param pdata Particle data arrays to integrate forward 1/2 step
    \param box The new box the particles where the particles now reside
    \param partial_scale Set to true to only scale those particles in the group
    \param box_len_scale Scaling factor by which to scale particle positions
    
    Scale all of the particle positions to fit inside the new box. ALL particles are scaled, not just those belonging
    to the group being integrated. Consequently, this kernel must be run with enough threads so that there is one
    thread for each particle in the box.
*/
extern "C" __global__ 
void gpu_npt_boxscale_kernel(gpu_pdata_arrays pdata,
                             gpu_boxsize box,
                             bool partial_scale,
                             float box_len_scale)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // scale ALL particles in the box
    if (idx < pdata.local_num)
        {
        // fetch particle position
        float4 pos = tex1Dfetch(pdata_pos_tex, idx);
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        if (!partial_scale)
            {
            px *= box_len_scale;
            py *= box_len_scale;
            pz *= box_len_scale;
            }
        
        // read in the image flags
        int4 image = tex1Dfetch(pdata_image_tex, idx);
        
        // fix periodic boundary conditions
        float x_shift = rintf(px * box.Lxinv);
        px -= box.Lx * x_shift;
        image.x += (int)x_shift;
        
        float y_shift = rintf(py * box.Lyinv);
        py -= box.Ly * y_shift;
        image.y += (int)y_shift;
        
        float z_shift = rintf(pz * box.Lzinv);
        pz -= box.Lz * z_shift;
        image.z += (int)z_shift;
        
        float4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;
        
        // write out the results
        pdata.pos[idx] = pos2;
        pdata.image[idx] = image;
        }
    }

/*! \param pdata Particle data arrays to integrate forward 1/2 step
    \param box The new box the particles where the particles now reside
    \param partial_scale Set to true to only scale those particles in the group
    \param Eta barostat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_boxscale_kernel(). See it for more details.
*/
cudaError_t gpu_npt_boxscale(const gpu_pdata_arrays &pdata,
                             const gpu_boxsize& box,
                             bool partial_scale,
                             float Eta,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (pdata.local_num / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    float box_len_scale = exp(Eta*deltaT);  // box length dilatation factor
    
    // scale the box before running the kernel
    gpu_boxsize scaled_box = box;
    scaled_box.Lx *= box_len_scale;
    scaled_box.Ly *= box_len_scale;
    scaled_box.Lz *= box_len_scale;
    scaled_box.Lxinv = 1.0f/scaled_box.Lx;
    scaled_box.Lyinv = 1.0f/scaled_box.Ly;
    scaled_box.Lzinv = 1.0f/scaled_box.Lz;

    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_image_tex, pdata.image, sizeof(int4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_npt_boxscale_kernel<<< grid, threads >>>(pdata, scaled_box, partial_scale, box_len_scale);
    
    return cudaSuccess;
    }

//! The texture for reading the net force
texture<float4, 1, cudaReadModeElementType> net_force_tex;

/*! \param pdata Particle data arrays to integrate forward 1/2 step
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param exp_v_fac exp_v_fac = \f$\exp(-\frac 1 4 (\eta+\xi)*\delta T)\f$ is the scaling factor for
velocity update and is a result of coupling to the thermo/barostat
    \param deltaT Time to advance (for one full step)
*/
extern "C" __global__ 
void gpu_npt_step_two_kernel(gpu_pdata_arrays pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float exp_v_fac,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        
        // read in the net force and compute the acceleration
        float4 accel = tex1Dfetch(net_force_tex, idx);
        float mass = tex1Dfetch(pdata_mass_tex, idx);
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        // fetch velocities
        float4 vel = tex1Dfetch(pdata_vel_tex, idx);
        
        // propagate velocities from t+1/2*deltaT to t+deltaT according to the
        // Nose-Hoover barostat
        vel.x = vel.x*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.x;
        vel.y = vel.y*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.y;
        vel.z = vel.z*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.z;
        
        // write out data
        pdata.vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx] = accel;
        }
    }

/*! \param pdata Particle Data to operate on
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param Xi theromstat variable in Nose-Hoover barostat
    \param Eta baromstat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_step_two(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             float Xi,
                             float Eta,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // precalulate velocity scaling factor due to Nose-Hoover barostat dynamics
    float exp_v_fac = exp(-1.0f/4.0f*(Eta+Xi)*deltaT);
    
    // bind the texture
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    // run the kernel
    gpu_npt_step_two_kernel<<< grid, threads >>>(pdata, d_group_members, group_size, exp_v_fac, deltaT);
    
    return cudaSuccess;
    }

// vim:syntax=cpp

