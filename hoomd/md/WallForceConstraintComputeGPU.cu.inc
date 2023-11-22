// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/WallForceConstraintComputeGPU.cuh"
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
gpu_compute_wall_friction<MANIFOLD_CLASS>(const unsigned int group_size,
					 unsigned int* d_index_array,
					 Scalar4* d_force,
					 Scalar* d_viral,
					 const Scalar4* d_pos,
					 const Scalar4* d_net_force,
					 MANIFOLD_CLASS manifold,
					 unsigned int block_size);

template hipError_t
gpu_compute_wall_constraint<MANIFOLD_CLASS>(const unsigned int group_size,
					 unsigned int* d_index_array,
					 Scalar4* d_force,
					 Scalar* d_viral,
					 const Scalar4* d_pos,
					 MANIFOLD_CLASS manifold,
					 unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
