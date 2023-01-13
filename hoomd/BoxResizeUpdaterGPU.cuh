// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#ifndef __BOX_RESIZE_UPDATER_GPU_CUH__
#define __BOX_RESIZE_UPDATER_GPU_CUH__

namespace hoomd
    {
namespace kernel
    {

hipError_t gpu_box_resize_updater(const unsigned int N,
                                Scalar4* d_pos,
                                const BoxDim& cur_box,
                                const BoxDim& new_box,
                                const unsigned int* d_group_members,
                                const unsigned int group_size,
                                int3* d_image);

    } // end namespace kernel
    } // end namespace hoomd

#endif // __BOX_RESIZE_UPDATER_GPU_CUH__
