// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: askeys

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#ifndef __FIRE_ENERGY_MINIMIZER_GPU_CUH__
#define __FIRE_ENERGY_MINIMIZER_GPU_CUH__

/*! \file FIREEnergyMinimizerGPU.cuh
    \brief Defines the interface to GPU kernel drivers used by FIREEnergyMinimizerGPU.
*/

//! Kernel driver for zeroing velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_zero_v(Scalar4 *d_vel,
                            unsigned int *d_group_members,
                            unsigned int group_size);

cudaError_t gpu_fire_zero_angmom(Scalar4 *d_angmom,
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

cudaError_t gpu_fire_compute_sum_all_angular(const unsigned int N,
                                    const Scalar4 *d_orientation,
                                    const Scalar3 *d_inertia,
                                    const Scalar4 *d_angmom,
                                    const Scalar4 *d_net_torque,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    Scalar* d_sum_all,
                                    Scalar* d_partial_sum_Pr,
                                    Scalar* d_partial_sum_wnorm,
                                    Scalar* d_partial_sum_tsq,
                                    unsigned int block_size,
                                    unsigned int num_blocks);

//! Kernel driver for updating the velocities called by FIREEnergyMinimizerGPU
cudaError_t gpu_fire_update_v(Scalar4 *d_vel,
                            const Scalar3 *d_accel,
                            unsigned int *d_group_members,
                            unsigned int group_size,
                            Scalar alpha,
                            Scalar factor_t);

cudaError_t gpu_fire_update_angmom(const Scalar4 *d_net_torque,
                              const Scalar4 *d_orientation,
                              const Scalar3 *d_inertia,
                              Scalar4 *d_angmom,
                              unsigned int *d_group_members,
                              unsigned int group_size,
                              Scalar alpha,
                              Scalar factor_r);

#endif //__FIRE_ENERGY_MINIMIZER_GPU_CUH__
