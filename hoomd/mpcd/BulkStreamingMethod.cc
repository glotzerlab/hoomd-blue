// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BulkStreamingMethod.h"

namespace hoomd
    {
mpcd::BulkStreamingMethod::BulkStreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                                               unsigned int cur_timestep,
                                               unsigned int period,
                                               int phase)
    : mpcd::BounceBackStreamingMethod<mpcd::detail::BulkGeometry>(
        sysdef,
        cur_timestep,
        period,
        phase,
        std::make_shared<mpcd::detail::BulkGeometry>())
    {
    }

void mpcd::detail::export_BulkStreamingMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::BulkStreamingMethod,
                     mpcd::StreamingMethod,
                     std::shared_ptr<mpcd::BulkStreamingMethod>>(m, "BulkStreamingMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, int>());
    }
    } // end namespace hoomd
