// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/BoxDeformerGPU.cuh
    \brief Declaration of CUDA kernels for md::BoxDeformerGPU
*/

#ifndef __BOX_DEFORMER_GPU_CUH__
#define __BOX_DEFORMER_GPU_CUH__

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for pbc wrapping after box deformation
hipError_t gpu_boxdeformer_wrap(const unsigned int N,
                                Scalar4* d_pos,
                                Scalar4* d_vel,
                                int3* d_image,
                                const BoxDim& new_box,
                                unsigned int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
