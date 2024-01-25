// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BounceBackStreamingMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::BounceBackStreamingMethodGPU
 *
 * \warning
 * This file needs separable compilation with ExternalFields.cu. Any plugins extending
 * the ConfinedStreamingGeometryGPU will also need to do separable compilation with
 * ExternalFields.cu.
 */

#include "BounceBackStreamingMethodGPU.cuh"
#include "StreamingGeometry.h"

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {

//! Template instantiation of parallel plate geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<mpcd::ParallelPlateGeometry>(const stream_args_t& args,
                                             const mpcd::ParallelPlateGeometry& geom);

//! Template instantiation of planar pore geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<mpcd::PlanarPoreGeometry>(const stream_args_t& args,
                                          const mpcd::PlanarPoreGeometry& geom);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
