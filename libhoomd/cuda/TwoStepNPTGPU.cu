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

#include "TwoStepNPTGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPTGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTGPU.
*/

//! Shared data used by NPT kernels for sum reductions
extern __shared__ float npt_sdata[];

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
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
void gpu_npt_step_one_kernel(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
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
        float4 pos = d_pos[idx];
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        // fetch particle velocity and acceleration
        float4 vel = d_vel[idx];
        Scalar3 accel = d_accel[idx];
        
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
        
        Scalar4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;
        
        // write out the results
        d_pos[idx] = pos2;
        d_vel[idx] = vel;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param partial_scale Set to true to only scale those particles in the group
    \param Xi theromstat variable in Nose-Hoover barostat
    \param Eta barostat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_step_one_kernel(). See it for more details.
*/
cudaError_t gpu_npt_step_one(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
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
    
    // precalculate scaling factors for baro/thermostat
    float exp_v_fac = exp(-1.0f/4.0f*(Eta+Xi)*deltaT);  // velocity scaling
    float exp_r_fac = exp(1.0f/2.0f*Eta*deltaT);        // position scaling
    
    // run the kernel
    gpu_npt_step_one_kernel<<< grid, threads >>>(d_pos,
                                                 d_vel,
                                                 d_accel,
                                                 d_group_members,
                                                 group_size,
                                                 partial_scale,
                                                 exp_v_fac,
                                                 exp_r_fac,
                                                 deltaT);
    
    return cudaSuccess;
    }
    
/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside
    \param partial_scale Set to true to only scale those particles in the group
    \param box_len_scale Scaling factor by which to scale particle positions
    
    Scale all of the particle positions to fit inside the new box. ALL particles are scaled, not just those belonging
    to the group being integrated. Consequently, this kernel must be run with enough threads so that there is one
    thread for each particle in the box.
*/
extern "C" __global__ 
void gpu_npt_boxscale_kernel(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             BoxDim box,
                             bool partial_scale,
                             float box_len_scale)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // scale ALL particles in the box
    if (idx < N)
        {
        // fetch particle position
        float4 postype = d_pos[idx];
        float3 pos = make_float3(postype.x, postype.y, postype.z);

        if (!partial_scale)
            {
            pos *= box_len_scale;
            }

        // read in the image flags
        int3 image = d_image[idx];

        // fix periodic boundary conditions
        box.wrap(pos, image);

        // write out the results
        d_pos[idx] = make_float4(pos.x, pos.y, pos.z, postype.w);
        d_image[idx] = image;
        }
    }

/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside
    \param partial_scale Set to true to only scale those particles in the group
    \param Eta barostat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_boxscale_kernel(). See it for more details.
*/
cudaError_t gpu_npt_boxscale(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             const BoxDim& box,
                             bool partial_scale,
                             float Eta,
                             float deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (N / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    float box_len_scale = exp(Eta*deltaT);  // box length dilatation factor
    
    // scale the box before running the kernel
    BoxDim scaled_box(box);
    scaled_box.setL(box.getL()*box_len_scale);

    // run the kernel
    gpu_npt_boxscale_kernel<<< grid, threads >>>(N, d_pos, d_image, scaled_box, partial_scale, box_len_scale);
    
    return cudaSuccess;
    }

/*! \param d_vel array of particle velocities and masses
    \param d_accel array of particle accelerations
    \param d_net_force array of net forces
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param exp_v_fac exp_v_fac = \f$\exp(-\frac 1 4 (\eta+\xi)*\delta T)\f$ is the scaling factor for
velocity update and is a result of coupling to the thermo/barostat
    \param deltaT Time to advance (for one full step)
*/
extern "C" __global__ 
void gpu_npt_step_two_kernel( Scalar4 *d_vel,
                              Scalar3 *d_accel,
                             const float4 *d_net_force,
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
        float4 accel = d_net_force[idx];

        // fetch velocities
        Scalar4 vel = d_vel[idx];

        float mass = vel.w;
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        // propagate velocities from t+1/2*deltaT to t+deltaT according to the
        // Nose-Hoover barostat
        vel.x = vel.x*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.x;
        vel.y = vel.y*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.y;
        vel.z = vel.z*exp_v_fac*exp_v_fac + (1.0f/2.0f)*deltaT*exp_v_fac*accel.z;
        
        // write out data
        d_vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = make_scalar3(accel.x, accel.y, accel.z);
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param Xi theromstat variable in Nose-Hoover barostat
    \param Eta baromstat variable in Nose-Hoover barostat
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
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
    
    // run the kernel
    gpu_npt_step_two_kernel<<< grid, threads >>>(d_vel, d_accel, d_net_force, d_group_members, group_size, exp_v_fac, deltaT);
    
    return cudaSuccess;
    }

// vim:syntax=cpp

