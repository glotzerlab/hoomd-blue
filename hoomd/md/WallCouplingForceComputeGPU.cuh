// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file WallCouplingForceComputeGPU.cuh
    \brief Declares GPU kernel code for calculating shear forces forces on the GPU. Used by
   WallCouplingForceComputeGPU.
*/

#ifndef __WALL_COUPLING_FORCE_COMPUTE_GPU_CUH__
#define __WALL_COUPLING_FORCE_COMPUTE_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_wall_coupling_force_set_forces(const unsigned int group_size,
                                                 unsigned int* d_index_array,
                                                 Scalar4* d_force,
                                                 const Scalar4* d_pos,
                                                 const Scalar r_f,
                                                 const Scalar t_f,
                                                 const Scalar l_f,
                                                 const Scalar R,
                                                 const unsigned int N,
                                                 unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
