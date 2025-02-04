// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/TwoStepRATTLEBDGPU.cuh"
#include "hoomd/md/TwoStepRATTLELangevinGPU.cuh"
#include "hoomd/md/TwoStepRATTLENVEGPU.cuh"

#include "hoomd/md/Manifold@_manifold_cpp@.h"

#define MANIFOLD_CLASS Manifold@_manifold_cpp@
// clang-format on

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template hipError_t
gpu_rattle_brownian_step_one<MANIFOLD_CLASS>(Scalar4* d_pos,
                                             int3* d_image,
                                             Scalar4* d_vel,
                                             const BoxDim& box,
                                             const unsigned int* d_tag,
                                             const unsigned int* d_group_members,
                                             const unsigned int group_size,
                                             const Scalar4* d_net_force,
                                             const Scalar3* d_gamma_r,
                                             Scalar4* d_orientation,
                                             Scalar4* d_torque,
                                             const Scalar3* d_inertia,
                                             Scalar4* d_angmom,
                                             const rattle_bd_step_one_args& rattle_bd_args,
                                             MANIFOLD_CLASS manifold,
                                             const bool aniso,
                                             const Scalar deltaT,
                                             const unsigned int D,
                                             const bool d_noiseless_t,
                                             const bool d_noiseless_r,
                                             const GPUPartition& gpu_partition);

template hipError_t
gpu_include_rattle_force_bd<MANIFOLD_CLASS>(const Scalar4* d_pos,
                                            Scalar4* d_net_force,
                                            Scalar* d_net_virial,
                                            const unsigned int* d_tag,
                                            const unsigned int* d_group_members,
                                            const unsigned int group_size,
                                            const rattle_bd_step_one_args& rattle_bd_args,
                                            MANIFOLD_CLASS manifold,
                                            size_t net_virial_pitch,
                                            const Scalar deltaT,
                                            const bool d_noiseless_t,
                                            const GPUPartition& gpu_partition);

template hipError_t gpu_rattle_langevin_step_two<MANIFOLD_CLASS>(
    const Scalar4* d_pos,
    Scalar4* d_vel,
    Scalar3* d_accel,
    const unsigned int* d_tag,
    unsigned int* d_group_members,
    unsigned int group_size,
    Scalar4* d_net_force,
    const rattle_langevin_step_two_args& rattle_langevin_args,
    MANIFOLD_CLASS manifold,
    Scalar deltaT,
    unsigned int D);

template hipError_t gpu_rattle_nve_step_two<MANIFOLD_CLASS>(Scalar4* d_pos,
                                                            Scalar4* d_vel,
                                                            Scalar3* d_accel,
                                                            unsigned int* d_group_members,
                                                            const GPUPartition& gpu_partition,
                                                            Scalar4* d_net_force,
                                                            MANIFOLD_CLASS manifold,
                                                            Scalar eta,
                                                            Scalar deltaT,
                                                            bool limit,
                                                            Scalar limit_val,
                                                            bool zero_force,
                                                            unsigned int block_size);

template hipError_t gpu_include_rattle_force_nve<MANIFOLD_CLASS>(const Scalar4* d_pos,
                                                                 const Scalar4* d_vel,
                                                                 Scalar3* d_accel,
                                                                 Scalar4* d_net_force,
                                                                 Scalar* d_net_virial,
                                                                 unsigned int* d_group_members,
                                                                 const GPUPartition& gpu_partition,
                                                                 size_t net_virial_pitch,
                                                                 MANIFOLD_CLASS manifold,
                                                                 Scalar eta,
                                                                 Scalar deltaT,
                                                                 bool zero_force,
                                                                 unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
