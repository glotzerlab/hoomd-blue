// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_
#define MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file mpcd/SlitGeometryFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SlitGeometryFillerGPU
 */

#include <cuda_runtime.h>

#include "SlitGeometry.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Draw virtual particles in the SlitGeometry
cudaError_t slit_draw_particles(Scalar4* d_pos,
                                Scalar4* d_vel,
                                unsigned int* d_tag,
                                const mpcd::detail::SlitGeometry& geom,
                                const Scalar z_min,
                                const Scalar z_max,
                                const BoxDim& box,
                                const Scalar mass,
                                const unsigned int type,
                                const unsigned int N_lo,
                                const unsigned int N_hi,
                                const unsigned int first_tag,
                                const unsigned int first_idx,
                                const Scalar kT,
                                const uint64_t timestep,
                                const uint16_t seed,
                                const unsigned int block_size);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_
