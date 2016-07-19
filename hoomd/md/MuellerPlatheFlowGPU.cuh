// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMPI.h"

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
                                        const unsigned int slab_direction,
                                        const unsigned int flow_direction,
                                        const unsigned int max_slab,
                                        const unsigned int min_slab,
                                        Scalar_Int*const last_max_vel,
                                        Scalar_Int*const last_min_vel,
                                        const bool has_max_slab,
                                        const bool has_min_slab,
                                        const unsigned int blocksize);

#endif//__MUELLER_PLATHE_FLOW_GPU_CUH__
