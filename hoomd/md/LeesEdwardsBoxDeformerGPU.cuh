// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file md/LeesEdwardsBoxDeformerGPU.cuh
    \brief Declaration of CUDA kernels for md::LeesEdwardsBoxDeformerGPU
*/

#ifndef __LEES_EDWARDS_BOX_DEFORMER_GPU_CUH__
#define __LEES_EDWARDS_BOX_DEFORMER_GPU_CUH__

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for box flipping and particle remapping
hipError_t gpu_lees_edwards_remap(const unsigned int N,
                                  Scalar4* d_pos,
                                  int3* d_image,
                                  const BoxDim& flipped_box,
                                  const Scalar xy,
                                  unsigned int block_size);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#endif
