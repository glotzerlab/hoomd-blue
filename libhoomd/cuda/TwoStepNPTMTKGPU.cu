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

#include "TwoStepNPTMTKGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file TwoStepNPTMTKGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU using the Martyna-Tobias-Klein update equations. Used by TwoStepNPTMTKGPU.
*/

//! Shared memory used in reducing the sum of the squared velocities
extern __shared__ Scalar npt_mtk_sdata[];

//! Kernel to propagate the positions and velocities, first half of NPT update
__global__ void gpu_npt_mtk_step_one_kernel(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             const Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar exp_thermo_fac,
                             Scalar mat_exp_v_xx,
                             Scalar mat_exp_v_xy,
                             Scalar mat_exp_v_xz,
                             Scalar mat_exp_v_yy,
                             Scalar mat_exp_v_yz,
                             Scalar mat_exp_v_zz,
                             Scalar mat_exp_v_int_xx,
                             Scalar mat_exp_v_int_xy,
                             Scalar mat_exp_v_int_xz,
                             Scalar mat_exp_v_int_yy,
                             Scalar mat_exp_v_int_yz,
                             Scalar mat_exp_v_int_zz,
                             Scalar mat_exp_r_xx,
                             Scalar mat_exp_r_xy,
                             Scalar mat_exp_r_xz,
                             Scalar mat_exp_r_yy,
                             Scalar mat_exp_r_yz,
                             Scalar mat_exp_r_zz,
                             Scalar mat_exp_r_int_xx,
                             Scalar mat_exp_r_int_xy,
                             Scalar mat_exp_r_int_xz,
                             Scalar mat_exp_r_int_yy,
                             Scalar mat_exp_r_int_yz,
                             Scalar mat_exp_r_int_zz,
                             Scalar deltaT,
                             bool rescale_all)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize eigenvectors
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // fetch particle position
        Scalar4 pos = d_pos[idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
        Scalar3 accel = d_accel[idx];;
        Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

        // apply thermostat update of velocity
        v *= exp_thermo_fac;

        // propagate velocity by half a time step and position by the full time step
        // by multiplying with upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        v.x += mat_exp_v_int_xx * accel.x + mat_exp_v_int_xy * accel.y + mat_exp_v_int_xz * accel.z;
        v.y += mat_exp_v_int_yy * accel.y + mat_exp_v_int_yz * accel.z;
        v.z += mat_exp_v_int_zz * accel.z;

        if (!rescale_all)
            {
            // rescale this group of particles
            r.x = mat_exp_r_xx * r.x + mat_exp_r_xy * r.y + mat_exp_r_xz * r.z;
            r.y = mat_exp_r_yy * r.y + mat_exp_r_yz * r.z;
            r.z = mat_exp_r_zz * r.z;
            }

        r.x += mat_exp_r_int_xx * v.x + mat_exp_r_int_xy * v.y + mat_exp_r_int_xz * v.z;
        r.y += mat_exp_r_int_yy * v.y + mat_exp_r_int_yz * v.z;
        r.z += mat_exp_r_int_zz * v.z;

        // write out the results
        d_pos[idx] = make_scalar4(r.x,r.y,r.z,pos.w);
        d_vel[idx] = make_scalar4(v.x,v.y,v.z,vel.w);
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param exp_thermo_fac Update factor for thermostat
    \param mat_exp_v Matrix exponential for velocity update
    \param mat_exp_v_int Integrated matrix exp for velocity update
    \param mat_exp_r Matrix exponential for position update
    \param mat_exp_r_int Integrated matrix exp for position update
    \param deltaT Time to advance (for one full step)
    \param deltaT Time to move forward in one whole step
    \param rescale_all True if all particles in the system should be rescaled at once

    This is just a kernel driver for gpu_npt_mtk_step_one_kernel(). See it for more details.
*/
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
                             bool rescale_all)
    {
    // setup the grid to run the kernel
    unsigned int block_size = 256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_npt_mtk_step_one_kernel<<< grid, threads >>>(d_pos,
                                                 d_vel,
                                                 d_accel,
                                                 d_group_members,
                                                 group_size,
                                                 exp_thermo_fac,
                                                 mat_exp_v[0],
                                                 mat_exp_v[1],
                                                 mat_exp_v[2],
                                                 mat_exp_v[3],
                                                 mat_exp_v[4],
                                                 mat_exp_v[5],
                                                 mat_exp_v_int[0],
                                                 mat_exp_v_int[1],
                                                 mat_exp_v_int[2],
                                                 mat_exp_v_int[3],
                                                 mat_exp_v_int[4],
                                                 mat_exp_v_int[5],
                                                 mat_exp_r[0],
                                                 mat_exp_r[1],
                                                 mat_exp_r[2],
                                                 mat_exp_r[3],
                                                 mat_exp_r[4],
                                                 mat_exp_r[5],
                                                 mat_exp_r_int[0],
                                                 mat_exp_r_int[1],
                                                 mat_exp_r_int[2],
                                                 mat_exp_r_int[3],
                                                 mat_exp_r_int[4],
                                                 mat_exp_r_int[5],
                                                 deltaT,
                                                 rescale_all);

    return cudaSuccess;
    }

/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    Wrap particle positions for all particles in the box
*/
extern "C" __global__
void gpu_npt_mtk_wrap_kernel(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             BoxDim box)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // wrap ALL particles in the box
    if (idx < N)
        {
        // fetch particle position
        Scalar4 postype = d_pos[idx];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        // read in the image flags
        int3 image = d_image[idx];

        // fix periodic boundary conditions
        box.wrap(pos, image);

        // write out the results
        d_pos[idx] = make_scalar4(pos.x, pos.y, pos.z, postype.w);
        d_image[idx] = image;
        }
    }

/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    This is just a kernel driver for gpu_npt_mtk_wrap_kernel(). See it for more details.
*/
cudaError_t gpu_npt_mtk_wrap(const unsigned int N,
                             Scalar4 *d_pos,
                             int3 *d_image,
                             const BoxDim& box)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (N / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_npt_mtk_wrap_kernel<<< grid, threads >>>(N, d_pos, d_image, box);

    return cudaSuccess;
    }

//! Kernel to propagate the positions and velocities, second half of NPT update
__global__ void gpu_npt_mtk_step_two_kernel(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             const Scalar4 *d_net_force,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar mat_exp_v_xx,
                             Scalar mat_exp_v_xy,
                             Scalar mat_exp_v_xz,
                             Scalar mat_exp_v_yy,
                             Scalar mat_exp_v_yz,
                             Scalar mat_exp_v_zz,
                             Scalar mat_exp_v_int_xx,
                             Scalar mat_exp_v_int_xy,
                             Scalar mat_exp_v_int_xz,
                             Scalar mat_exp_v_int_yy,
                             Scalar mat_exp_v_int_yz,
                             Scalar mat_exp_v_int_zz,
                             Scalar deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];

        // compute acceleration
        Scalar minv = Scalar(1.0)/vel.w;
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
        accel *= minv;

        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);

        // propagate velocity by half a time step by multiplying with an upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        v.x += mat_exp_v_int_xx * accel.x + mat_exp_v_int_xy * accel.y + mat_exp_v_int_xz * accel.z;
        v.y += mat_exp_v_int_yy * accel.y + mat_exp_v_int_yz * accel.z;
        v.z += mat_exp_v_int_zz * accel.z;

        // write out velocity
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);

        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param mat_exp_v Matrix exponential for velocity update
    \param mat_exp_v_int Integrated matrix exp for velocity update
    \param d_net_force Net force on each particle

    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_mtk_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_mtk_step_two(Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar4 *d_net_force,
                             Scalar* mat_exp_v,
                             Scalar* mat_exp_v_int,
                             Scalar deltaT)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_npt_mtk_step_two_kernel<<< grid, threads >>>(d_vel,
                                                     d_accel,
                                                     d_net_force,
                                                     d_group_members,
                                                     group_size,
                                                     mat_exp_v[0],
                                                     mat_exp_v[1],
                                                     mat_exp_v[2],
                                                     mat_exp_v[3],
                                                     mat_exp_v[4],
                                                     mat_exp_v[5],
                                                     mat_exp_v_int[0],
                                                     mat_exp_v_int[1],
                                                     mat_exp_v_int[2],
                                                     mat_exp_v_int[3],
                                                     mat_exp_v_int[4],
                                                     mat_exp_v_int[5],
                                                     deltaT);

    return cudaSuccess;
    }

//! GPU kernel to perform partial reduction of temperature
__global__ void gpu_npt_mtk_temperature_partial(unsigned int *d_group_members,
                                                unsigned int group_size,
                                                Scalar *d_scratch,
                                                Scalar4 *d_velocity)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar mv2_element; // element of scratch space read in
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        Scalar4 vel = d_velocity[idx];
        Scalar mass = vel.w;

        mv2_element =  mass * (vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        }
    else
        {
        // non-participating thread: contribute 0 to the sum
        mv2_element = Scalar(0.0);
        }

    npt_mtk_sdata[threadIdx.x] = mv2_element;
    __syncthreads();

    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            npt_mtk_sdata[threadIdx.x] += npt_mtk_sdata[threadIdx.x + offs];

        offs >>= 1;
        __syncthreads();
        }

    // write out partial sum
    if (threadIdx.x == 0)
        d_scratch[blockIdx.x] = npt_mtk_sdata[0];

     }

//! GPU kernel to perform final reduction of temperature
__global__ void gpu_npt_mtk_temperature_final_sum(Scalar *d_scratch,
                                                  Scalar *d_temperature,
                                                  unsigned int ndof,
                                                  unsigned int num_partial_sums)
    {
    Scalar final_sum(0.0);

    for (int start = 0; start < num_partial_sums; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_partial_sums)
            {
            npt_mtk_sdata[threadIdx.x] = d_scratch[start + threadIdx.x];
            }
        else
            npt_mtk_sdata[threadIdx.x] = Scalar(0.0);

        __syncthreads();

        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                npt_mtk_sdata[threadIdx.x] += npt_mtk_sdata[threadIdx.x + offs];

            offs >>=1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            final_sum += npt_mtk_sdata[0];
        }

    if (threadIdx.x == 0)
        *d_temperature = final_sum/Scalar(ndof);
    }

/*!\param d_temperature Device variable to store the temperature value (output)
   \param d_vel Array of particle velocities and masses
   \param d_scratch Temporary scratch space for reduction
   \param num_blocks Number of CUDA blocks used in reduction
   \param block_size Size of blocks used in reduction
   \param d_group_members Members of group for which the reduction is performed
   \param group_size Size of group
   \param ndof Number of degrees of freedom of group

   This function performs the reduction of the temperature on the GPU. It is just
   a driver function that calls the appropriate GPU kernels.
   */
cudaError_t gpu_npt_mtk_temperature(Scalar *d_temperature,
                                    Scalar4 *d_vel,
                                    Scalar *d_scratch,
                                    unsigned int num_blocks,
                                    unsigned int block_size,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    unsigned int ndof)
    {
    assert(d_temperature);
    assert(d_vel);
    assert(d_scratch);

    dim3 grid(num_blocks,1,1);
    dim3 threads(block_size,1,1);

    unsigned int shared_bytes = sizeof(Scalar)*block_size;

    // reduce squared velocity norm times mass, first pass
    gpu_npt_mtk_temperature_partial<<<grid, threads, shared_bytes>>>(
                                                d_group_members,
                                                group_size,
                                                d_scratch,
                                                d_vel);


    unsigned int final_block_size = 512;
    grid = dim3(1,1,1);
    threads = dim3(final_block_size, 1, 1);
    shared_bytes = sizeof(Scalar)*final_block_size;

    // reduction, second pass
    gpu_npt_mtk_temperature_final_sum<<<grid, threads, shared_bytes>>>(
                                                d_scratch,
                                                d_temperature,
                                                ndof,
                                                num_blocks);

    return cudaSuccess;
    }

/*! \param d_vel array of particle velocities and masses
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param exp_v_fac_thermo scaling factor (per direction) for velocity update generated by thermostat

    GPU kernel to thermostat velocities
*/
__global__ void gpu_npt_mtk_thermostat_kernel(Scalar4 *d_vel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar exp_v_fac_thermo)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);

        // rescale
        v *= exp_v_fac_thermo;

        // write out the results
        d_vel[idx] = make_scalar4(v.x,v.y,v.z,vel.w);
        }
    }

/*! \param d_vel array of particle velocities
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param xi Thermostat velocity
    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_step_kernel(). See it for more details.
*/
cudaError_t gpu_npt_mtk_thermostat(Scalar4 *d_vel,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             Scalar exp_v_fac_thermo)
    {
    // setup the grid to run the kernel
    unsigned int block_size=256;
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_npt_mtk_thermostat_kernel<<< grid, threads >>>(d_vel,
                                                     d_group_members,
                                                     group_size,
                                                     exp_v_fac_thermo);

    return cudaSuccess;
    }

__global__ void gpu_npt_mtk_rescale_kernel(unsigned int N,
                                           Scalar4 *d_postype,
                                           Scalar mat_exp_r_xx,
                                           Scalar mat_exp_r_xy,
                                           Scalar mat_exp_r_xz,
                                           Scalar mat_exp_r_yy,
                                           Scalar mat_exp_r_yz,
                                           Scalar mat_exp_r_zz)
    {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // rescale position
    Scalar4 postype = d_postype[idx];
    Scalar3 r = make_scalar3(postype.x,postype.y,postype.z);

    r.x = mat_exp_r_xx * r.x + mat_exp_r_xy * r.y + mat_exp_r_xz * r.z;
    r.y = mat_exp_r_yy * r.y + mat_exp_r_yz* r.z;
    r.z = mat_exp_r_zz * r.z;

    d_postype[idx] = make_scalar4(r.x, r.y, r.z, postype.w);
    }

void gpu_npt_mtk_rescale(unsigned int N,
                       Scalar4 *d_postype,
                       Scalar mat_exp_r_xx,
                       Scalar mat_exp_r_xy,
                       Scalar mat_exp_r_xz,
                       Scalar mat_exp_r_yy,
                       Scalar mat_exp_r_yz,
                       Scalar mat_exp_r_zz)
    {
    unsigned int block_size = 256;

    dim3 grid( (N / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    gpu_npt_mtk_rescale_kernel<<<grid, threads>>> (N,
        d_postype,
        mat_exp_r_xx,
        mat_exp_r_xy,
        mat_exp_r_xz,
        mat_exp_r_yy,
        mat_exp_r_yz,
        mat_exp_r_zz);
    }
