// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ConfinedStreamingMethodGPU.cu
 * \brief Defines GPU functions and kernels used by mpcd::ConfinedStreamingMethodGPU
 *
 * \warning
 * This file needs separable compilation with ExternalFields.cu. Any plugins extending
 * the ConfinedStreamingGeometryGPU will also need to do separable compilation with
 * ExternalFields.cu.
 */

#include "ConfinedStreamingMethodGPU.cuh"
#include "StreamingGeometry.h"

#include "ExternalField.h"
#include "hoomd/GPUPolymorph.cuh"

namespace hoomd
    {
namespace mpcd
    {
namespace gpu
    {
//! Template instantiation of bulk geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<mpcd::detail::BulkGeometry>(const stream_args_t& args,
                                            const mpcd::detail::BulkGeometry& geom);

//! Template instantiation of slit geometry streaming
template cudaError_t __attribute__((visibility("default")))
confined_stream<mpcd::detail::SlitGeometry>(const stream_args_t& args,
                                            const mpcd::detail::SlitGeometry& geom);

//! Template instantiation of slit geometry streaming
template cudaError_t
confined_stream<mpcd::detail::SlitPoreGeometry>(const stream_args_t& args,
                                                const mpcd::detail::SlitPoreGeometry& geom);

    } // end namespace gpu
    } // end namespace mpcd
    } // end namespace hoomd
