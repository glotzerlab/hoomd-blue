// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ShearForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating shear forces forces on the GPU. Used by
   ShearForceComputeGPU.
*/

#ifndef __SHEAR_FORCE_COMPUTE_GPU_CUH__
#define __SHEAR_FORCE_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_shear_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 const Scalar4* d_pos,
                                                 const Scalar s_f,
                                                 const unsigned int N,
                                                 unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
