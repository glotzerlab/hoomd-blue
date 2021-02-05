// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepRATTLEBDGPU.cuh"
#include "TwoStepRATTLELangevinGPU.cuh"
#include "TwoStepRATTLENVEGPU.cuh"

#include "ManifoldClassEllipsoid.h"


template hipError_t gpu_include_rattle_force_bd<ManifoldClassEllipsoid>(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar4 *d_net_force,
                                  Scalar3 *d_f_brownian,
                                  Scalar *d_net_virial,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_rtag,
                                  const unsigned int *d_groupTags,
                                  const unsigned int group_size,
                                  const rattle_bd_step_one_args& rattle_bd_args,
			          ManifoldClassEllipsoid manifold,
                                  unsigned int net_virial_pitch,
                                  const Scalar deltaT,
                                  const bool d_noiseless_t,
                                  const GPUPartition& gpu_partition
                                  );

template hipError_t gpu_rattle_langevin_step_two<ManifoldClassEllipsoid>(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar3 *d_accel,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar4 *d_net_force,
                                  const rattle_langevin_step_two_args& rattle_langevin_args,
                                  ManifoldClassEllipsoid manifold,
                                  Scalar deltaT,
                                  unsigned int D);

template hipError_t gpu_rattle_nve_step_two<ManifoldClassEllipsoid>(Scalar4 *d_pos,
                             Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             Scalar4 *d_net_force,
                             ManifoldClassEllipsoid manifold,
                             Scalar eta,
                             Scalar deltaT,
                             bool limit,
                             Scalar limit_val,
                             bool zero_force,
                             unsigned int block_size);

template hipError_t gpu_include_rattle_force_nve<ManifoldClassEllipsoid>(const Scalar4 *d_pos,
                             const Scalar4 *d_vel,
                             Scalar3 *d_accel,
                             Scalar4 *d_net_force,
                             Scalar *d_net_virial,
                             unsigned int *d_group_members,
                             const GPUPartition& gpu_partition,
                             unsigned int net_virial_pitch,
			     ManifoldClassEllipsoid manifold,
                             Scalar eta,
                             Scalar deltaT,
                             bool zero_force,
                             unsigned int block_size);
