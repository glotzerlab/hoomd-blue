// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BulkStreamingMethod.h
 * \brief Declaration of mpcd::BulkStreamingMethod
 */

#ifndef MPCD_BULK_STREAMING_METHOD_H_
#define MPCD_BULK_STREAMING_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BounceBackStreamingMethod.h"
#include "BulkGeometry.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {

class PYBIND11_EXPORT BulkStreamingMethod
    : public BounceBackStreamingMethod<mpcd::detail::BulkGeometry>
    {
    public:
    BulkStreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                        unsigned int cur_timestep,
                        unsigned int period,
                        int phase);
    };

namespace detail
    {
//! Export mpcd::BulkStreamingMethod to python
/*!
 * \param m Python module to export to
 */
void export_BulkStreamingMethod(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_BULK_STREAMING_METHOD_H_
