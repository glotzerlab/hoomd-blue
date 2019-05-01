// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file MuellerPlatheFlowGPU.cuh
    \brief Declares GPU kernel code for calculating MinMax velocities and updates for the flow.
*/

#ifndef __MUELLER_PLATHE_FLOW_GPU_CUH__
#define __MUELLER_PLATHE_FLOW_GPU_CUH__

cudaError_t gpu_search_min_max_velocity(const unsigned int group_size,
                                        const Scalar4*const d_vel,
                                        const Scalar4*const d_pos,
                                        const unsigned int *const d_tag,
                                        const unsigned int *const d_rtag,
                                        const unsigned int *const d_group_members,
                                        const BoxDim gl_box,
                                        const unsigned int Nslabs,
                                        const unsigned int max_slab,
                                        const unsigned int min_slab,
                                        Scalar3*const last_max_vel,
                                        Scalar3*const last_min_vel,
                                        const bool has_max_slab,
                                        const bool has_min_slab,
                                        const unsigned int blocksize,
                                        flow_enum::Direction flow_direction,
                                        flow_enum::Direction slab_direction);

cudaError_t gpu_update_min_max_velocity(const unsigned int *const d_rtag,
                                        Scalar4*const d_vel,
                                        const unsigned int Ntotal,
                                        const Scalar3 last_max_vel,
                                        const Scalar3 last_min_vel,
                                        const flow_enum::Direction flow_direction);
#endif//__MUELLER_PLATHE_FLOW_GPU_CUH__
