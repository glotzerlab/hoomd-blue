// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file CollectiveActiveForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating collective active forces on the GPU. Used by
   CollectiveActiveForceComputeGPU.
*/

#ifndef __COLLECTIVE_ACTIVE_FORCE_COMPUTE_GPU_CUH__
#define __COLLECTIVE_ACTIVE_FORCE_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_collective_active_force_set_forces(const unsigned int group_size,
                                                          unsigned int* d_index_array,
                                                          Scalar4* d_force,
                                                          const Scalar4* d_pos,
                                                          const Scalar4* d_f_act,
                                                          unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
