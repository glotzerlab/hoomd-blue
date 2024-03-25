// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepConstantPressureGPU.cuh"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file TwoStepNPTMTKGPU.cu
    \brief Defines GPU kernel code for NPT integration on the GPU using the Martyna-Tobias-Klein
   update equations. Used by TwoStepNPTMTKGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel to propagate the positions and velocities, first half of NPT update
__global__ void gpu_npt_mtk_step_one_kernel(Scalar4* d_pos,
                                            Scalar4* d_vel,
                                            const Scalar3* d_accel,
                                            unsigned int* d_group_members,
                                            const unsigned int nwork,
                                            const unsigned int offset,
                                            Scalar thermo_rescale,
                                            Scalar mat_exp_v_xx,
                                            Scalar mat_exp_v_xy,
                                            Scalar mat_exp_v_xz,
                                            Scalar mat_exp_v_yy,
                                            Scalar mat_exp_v_yz,
                                            Scalar mat_exp_v_zz,
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
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize eigenvectors
    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        // fetch particle position
        Scalar4 pos = d_pos[idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];
        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);
        Scalar3 accel = d_accel[idx];
        ;
        Scalar3 r = make_scalar3(pos.x, pos.y, pos.z);

        // advance velocity
        v += deltaT / Scalar(2.0) * accel;

        // propagate velocity by half a time step and position by the full time step
        // by multiplying with upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        // apply thermostat update of velocity
        v *= thermo_rescale;

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
        d_pos[idx] = make_scalar4(r.x, r.y, r.z, pos.w);
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param thermo_rescale Update factor for thermostat
    \param mat_exp_v Matrix exponential for velocity update
    \param mat_exp_r Matrix exponential for position update
    \param mat_exp_r_int Integrated matrix exp for position update
    \param deltaT Time to advance (for one full step)
    \param deltaT Time to move forward in one whole step
    \param rescale_all True if all particles in the system should be rescaled at once

    This is just a kernel driver for gpu_npt_mtk_step_one_kernel(). See it for more details.
*/
hipError_t gpu_npt_rescale_step_one(Scalar4* d_pos,
                                    Scalar4* d_vel,
                                    const Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const GPUPartition& gpu_partition,
                                    Scalar thermo_rescale,
                                    Scalar* mat_exp_v,
                                    Scalar* mat_exp_r,
                                    Scalar* mat_exp_r_int,
                                    Scalar deltaT,
                                    bool rescale_all,
                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_npt_mtk_step_one_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_npt_mtk_step_one_kernel),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           d_pos,
                           d_vel,
                           d_accel,
                           d_group_members,
                           nwork,
                           range.first,
                           thermo_rescale,
                           mat_exp_v[0],
                           mat_exp_v[1],
                           mat_exp_v[2],
                           mat_exp_v[3],
                           mat_exp_v[4],
                           mat_exp_v[5],
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
        }

    return hipSuccess;
    }

/*! \param N number of particles in the system
    \param d_pos array of particle positions
    \param d_image array of particle images
    \param box The new box the particles where the particles now reside

    Wrap particle positions for all particles in the box
*/
__global__ void gpu_npt_mtk_wrap_kernel(const unsigned int nwork,
                                        const unsigned int offset,
                                        Scalar4* d_pos,
                                        int3* d_image,
                                        BoxDim box)
    {
    // determine which particle this thread works on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // wrap ALL particles in the box
    if (idx < nwork)
        {
        idx += offset;

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
hipError_t gpu_npt_rescale_wrap(const GPUPartition& gpu_partition,
                                Scalar4* d_pos,
                                int3* d_image,
                                const BoxDim& box,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_npt_mtk_wrap_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_npt_mtk_wrap_kernel),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           nwork,
                           range.first,
                           d_pos,
                           d_image,
                           box);
        }

    return hipSuccess;
    }

//! Kernel to propagate the positions and velocities, second half of NPT update
__global__ void gpu_npt_mtk_step_two_kernel(Scalar4* d_vel,
                                            Scalar3* d_accel,
                                            const Scalar4* d_net_force,
                                            unsigned int* d_group_members,
                                            const unsigned int nwork,
                                            const unsigned int offset,
                                            Scalar mat_exp_v_xx,
                                            Scalar mat_exp_v_xy,
                                            Scalar mat_exp_v_xz,
                                            Scalar mat_exp_v_yy,
                                            Scalar mat_exp_v_yz,
                                            Scalar mat_exp_v_zz,
                                            Scalar deltaT,
                                            Scalar thermo_rescale)
    {
    // determine which particle this thread works on
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx + offset;
        unsigned int idx = d_group_members[group_idx];

        // fetch particle velocity and acceleration
        Scalar4 vel = d_vel[idx];

        // compute acceleration
        Scalar minv = Scalar(1.0) / vel.w;
        Scalar4 net_force = d_net_force[idx];
        Scalar3 accel = make_scalar3(net_force.x, net_force.y, net_force.z);
        accel *= minv;

        Scalar3 v = make_scalar3(vel.x, vel.y, vel.z);

        // apply thermostat rescaling
        v = v * thermo_rescale;

        // propagate velocity by half a time step by multiplying with an upper triangular matrix
        v.x = mat_exp_v_xx * v.x + mat_exp_v_xy * v.y + mat_exp_v_xz * v.z;
        v.y = mat_exp_v_yy * v.y + mat_exp_v_yz * v.z;
        v.z = mat_exp_v_zz * v.z;

        // advance velocity
        v += deltaT / Scalar(2.0) * accel;

        // write out velocity
        d_vel[idx] = make_scalar4(v.x, v.y, v.z, vel.w);

        // since we calculate the acceleration, we need to write it for the next step
        d_accel[idx] = accel;
        }
    }

/*! \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param mat_exp_v Matrix exponential for velocity update
    \param d_net_force Net force on each particle

    \param deltaT Time to move forward in one whole step

    This is just a kernel driver for gpu_npt_mtk_step_kernel(). See it for more details.
*/
hipError_t gpu_npt_rescale_step_two(Scalar4* d_vel,
                                    Scalar3* d_accel,
                                    unsigned int* d_group_members,
                                    const GPUPartition& gpu_partition,
                                    Scalar4* d_net_force,
                                    Scalar* mat_exp_v,
                                    Scalar deltaT,
                                    Scalar thermo_rescale,
                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_npt_mtk_step_two_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        // run the kernel
        hipLaunchKernelGGL((gpu_npt_mtk_step_two_kernel),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           d_vel,
                           d_accel,
                           d_net_force,
                           d_group_members,
                           nwork,
                           range.first,
                           mat_exp_v[0],
                           mat_exp_v[1],
                           mat_exp_v[2],
                           mat_exp_v[3],
                           mat_exp_v[4],
                           mat_exp_v[5],
                           deltaT,
                           thermo_rescale);
        }

    return hipSuccess;
    }

__global__ void gpu_npt_mtk_rescale_kernel(const unsigned int nwork,
                                           const unsigned int offset,
                                           Scalar4* d_postype,
                                           Scalar mat_exp_r_xx,
                                           Scalar mat_exp_r_xy,
                                           Scalar mat_exp_r_xz,
                                           Scalar mat_exp_r_yy,
                                           Scalar mat_exp_r_yz,
                                           Scalar mat_exp_r_zz)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nwork)
        return;
    idx += offset;

    // rescale position
    Scalar4 postype = d_postype[idx];
    Scalar3 r = make_scalar3(postype.x, postype.y, postype.z);

    r.x = mat_exp_r_xx * r.x + mat_exp_r_xy * r.y + mat_exp_r_xz * r.z;
    r.y = mat_exp_r_yy * r.y + mat_exp_r_yz * r.z;
    r.z = mat_exp_r_zz * r.z;

    d_postype[idx] = make_scalar4(r.x, r.y, r.z, postype.w);
    }

void gpu_npt_rescale_rescale(const GPUPartition& gpu_partition,
                             Scalar4* d_postype,
                             Scalar mat_exp_r_xx,
                             Scalar mat_exp_r_xy,
                             Scalar mat_exp_r_xz,
                             Scalar mat_exp_r_yy,
                             Scalar mat_exp_r_yz,
                             Scalar mat_exp_r_zz,
                             const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_npt_mtk_rescale_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;

        // setup the grid to run the kernel
        dim3 grid((nwork / run_block_size) + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        hipLaunchKernelGGL((gpu_npt_mtk_rescale_kernel),
                           dim3(grid),
                           dim3(threads),
                           0,
                           0,
                           nwork,
                           range.first,
                           d_postype,
                           mat_exp_r_xx,
                           mat_exp_r_xy,
                           mat_exp_r_xz,
                           mat_exp_r_yy,
                           mat_exp_r_yz,
                           mat_exp_r_zz);
        }
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
