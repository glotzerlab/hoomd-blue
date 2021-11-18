// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "ActiveForceConstraintComputeGPU.cuh"
#include "ManifoldEllipsoid.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template hipError_t
gpu_compute_active_force_set_constraints<ManifoldEllipsoid>(const unsigned int group_size,
                                                            unsigned int* d_index_array,
                                                            const Scalar4* d_pos,
                                                            Scalar4* d_orientation,
                                                            const Scalar4* d_f_act,
                                                            ManifoldEllipsoid manifold,
                                                            unsigned int block_size);

template hipError_t gpu_compute_active_force_constraint_rotational_diffusion<ManifoldEllipsoid>(
    const unsigned int group_size,
    unsigned int* d_tag,
    unsigned int* d_index_array,
    const Scalar4* d_pos,
    Scalar4* d_orientation,
    ManifoldEllipsoid manifold,
    bool is2D,
    const Scalar rotationDiff,
    const uint64_t timestep,
    const uint16_t seed,
    unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
