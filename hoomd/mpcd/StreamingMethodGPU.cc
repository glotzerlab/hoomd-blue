// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingMethodGPU.cc
 * \brief Definition of mpcd::StreamingMethodGPU
 */

#include "StreamingMethodGPU.h"
#include "StreamingMethodGPU.cuh"

/*!
 * \param sysdata MPCD system data
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 */
mpcd::StreamingMethodGPU::StreamingMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                             unsigned int cur_timestep,
                                             unsigned int period,
                                             int phase)
    : mpcd::StreamingMethod(sysdata, cur_timestep, period, phase)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_stream", m_exec_conf));
    }

/*!
 * \param timestep Current time to stream
 */
void mpcd::StreamingMethodGPU::stream(unsigned int timestep)
    {
    if (!shouldStream(timestep)) return;

    if (m_prof) m_prof->push(m_exec_conf, "MPCD stream");

    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::read);

    m_tuner->begin();
    mpcd::gpu::stream(d_pos.data,
                      d_vel.data,
                      m_pdata->getBox(),
                      m_mpcd_dt,
                      m_mpcd_pdata->getN(),
                      m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();

    // particles have moved, so the cell cache is no longer valid
    m_mpcd_pdata->invalidateCellCache();
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void mpcd::detail::export_StreamingMethodGPU(pybind11::module& m)
    {
    namespace py = pybind11;

    py::class_<mpcd::StreamingMethodGPU, std::shared_ptr<mpcd::StreamingMethodGPU> >
        (m, "StreamingMethodGPU", py::base<mpcd::StreamingMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int>());
    }
