// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

template<class Force>
class PYBIND11_EXPORT BulkStreamingMethodGPU
    : public BounceBackStreamingMethodGPU<mpcd::detail::BulkGeometry, Force>
    {
    public:
    BulkStreamingMethodGPU(std::shared_ptr<SystemDefinition> sysdef,
                           unsigned int cur_timestep,
                           unsigned int period,
                           int phase,
                           std::shared_ptr<Force> force)
        : mpcd::BounceBackStreamingMethodGPU<mpcd::detail::BulkGeometry, Force>(
              sysdef,
              cur_timestep,
              period,
              phase,
              std::make_shared<mpcd::detail::BulkGeometry>(),
              force)
        {
        }
    };

namespace detail
    {
//! Export mpcd::BulkStreamingMethodGPU to python
/*!
 * \param m Python module to export to
 */
template<class Force> void export_BulkStreamingMethodGPU(pybind11::module& m)
    {
    const std::string name = "BulkStreamingMethod" + Force::getName() + "GPU";
    pybind11::class_<mpcd::BulkStreamingMethodGPU<Force>,
                     mpcd::StreamingMethod,
                     std::shared_ptr<mpcd::BulkStreamingMethodGPU<Force>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            unsigned int,
                            unsigned int,
                            int,
                            std::shared_ptr<Force>>())
        .def_property_readonly("force", &mpcd::BulkStreamingMethodGPU<Force>::getForce);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_BULK_STREAMING_METHOD_GPU_H_
