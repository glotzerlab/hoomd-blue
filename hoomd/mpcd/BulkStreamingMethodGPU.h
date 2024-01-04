// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkStreamingMethodGPU.h
 * \brief Declaration of mpcd::BulkStreamingMethodGPU
 */

#ifndef MPCD_BULK_STREAMING_METHOD_GPU_H_
#define MPCD_BULK_STREAMING_METHOD_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BounceBackStreamingMethodGPU.h"
#include "BulkGeometry.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {

class PYBIND11_EXPORT BulkStreamingMethodGPU
    : public BounceBackStreamingMethodGPU<mpcd::detail::BulkGeometry>
    {
    public:
    BulkStreamingMethodGPU(std::shared_ptr<SystemDefinition> sysdef,
                           unsigned int cur_timestep,
                           unsigned int period,
                           int phase);
    };

namespace detail
    {
//! Export mpcd::BulkStreamingMethodGPU to python
/*!
 * \param m Python module to export to
 */
void export_BulkStreamingMethodGPU(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_BULK_STREAMING_METHOD_GPU_H_
