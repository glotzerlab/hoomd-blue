// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

template<class Force>
class PYBIND11_EXPORT BulkStreamingMethod
    : public BounceBackStreamingMethod<mpcd::detail::BulkGeometry, Force>
    {
    public:
    BulkStreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                        unsigned int cur_timestep,
                        unsigned int period,
                        int phase,
                        std::shared_ptr<Force> force)
        : mpcd::BounceBackStreamingMethod<mpcd::detail::BulkGeometry, Force>(
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
//! Export mpcd::BulkStreamingMethod to python
/*!
 * \param m Python module to export to
 */
template<class Force> void export_BulkStreamingMethod(pybind11::module& m)
    {
    const std::string name = "BulkStreamingMethod" + Force::getName();
    pybind11::class_<mpcd::BulkStreamingMethod<Force>,
                     mpcd::StreamingMethod,
                     std::shared_ptr<mpcd::BulkStreamingMethod<Force>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            unsigned int,
                            unsigned int,
                            int,
                            std::shared_ptr<Force>>())
        .def_property_readonly("mpcd_particle_force", &mpcd::BulkStreamingMethod<Force>::getForce);
    }

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd

#endif // MPCD_BULK_STREAMING_METHOD_H_
