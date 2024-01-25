// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BulkStreamingMethodGPU.h"

namespace hoomd
    {
mpcd::BulkStreamingMethodGPU::BulkStreamingMethodGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                     unsigned int cur_timestep,
                                                     unsigned int period,
                                                     int phase)
    : mpcd::BounceBackStreamingMethodGPU<mpcd::detail::BulkGeometry>(
        sysdef,
        cur_timestep,
        period,
        phase,
        std::make_shared<mpcd::detail::BulkGeometry>())
    {
    }

void mpcd::detail::export_BulkStreamingMethodGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::BulkStreamingMethodGPU,
                     mpcd::StreamingMethod,
                     std::shared_ptr<mpcd::BulkStreamingMethodGPU>>(m, "BulkStreamingMethodGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, int>());
    }
    } // end namespace hoomd
