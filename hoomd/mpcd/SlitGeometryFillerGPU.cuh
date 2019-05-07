// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_
#define MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_

/*!
 * \file mpcd/SlitGeometryFillerGPU.cuh
 * \brief Declaration of CUDA kernels for mpcd::SlitGeometryFillerGPU
 */

#include <cuda_runtime.h>

#include "SlitGeometry.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

namespace mpcd
{
namespace gpu
{

//! Draw virtual particles in the SlitGeometry
cudaError_t slit_draw_particles(Scalar4 *d_pos,
                                Scalar4 *d_vel,
                                unsigned int *d_tag,
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
                                const unsigned int timestep,
                                const unsigned int seed,
                                const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_SLIT_GEOMETRY_FILLER_GPU_CUH_
