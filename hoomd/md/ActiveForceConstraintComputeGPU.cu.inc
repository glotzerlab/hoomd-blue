// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// See md/CMakeLists.txt for the source of these variables to be processed by CMake's
// configure_file().

// clang-format off
#include "hoomd/md/ActiveForceConstraintComputeGPU.cuh"
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
gpu_compute_active_force_set_constraints<MANIFOLD_CLASS>(const unsigned int group_size,
                                                         unsigned int* d_index_array,
                                                         const Scalar4* d_pos,
                                                         Scalar4* d_orientation,
                                                         const Scalar4* d_f_act,
                                                         MANIFOLD_CLASS manifold,
                                                         unsigned int block_size);

template hipError_t gpu_compute_active_force_constraint_rotational_diffusion<MANIFOLD_CLASS>(
    const unsigned int group_size,
    unsigned int* d_tag,
    unsigned int* d_index_array,
    const Scalar4* d_pos,
    Scalar4* d_orientation,
    MANIFOLD_CLASS manifold,
    bool is2D,
    const Scalar rotationDiff,
    const uint64_t timestep,
    const uint16_t seed,
    unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
