// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SlitPoreGeometryFillerGPU.cc
 * \brief Definition of mpcd::SlitPoreGeometryFillerGPU
 */

#include "SlitPoreGeometryFillerGPU.h"
#include "SlitPoreGeometryFillerGPU.cuh"

mpcd::SlitPoreGeometryFillerGPU::SlitPoreGeometryFillerGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                   Scalar density,
                                                   unsigned int type,
                                                   std::shared_ptr<::Variant> T,
                                                   unsigned int seed,
                                                   std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom)
    : mpcd::SlitPoreGeometryFiller(sysdata, density, type, T, seed, geom)
    {
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_slit_filler", m_exec_conf));
    }

/*!
 * \param timestep Current timestep
 */
void mpcd::SlitPoreGeometryFillerGPU::drawParticles(unsigned int timestep)
    {
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(), access_location::device, access_mode::readwrite);

    // boxes for filling
    ArrayHandle<Scalar4> d_boxes(m_boxes, access_location::device, access_mode::read);
    ArrayHandle<uint2> d_ranges(m_ranges, access_location::device, access_mode::read);

    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;

    m_tuner->begin();
    mpcd::gpu::slit_pore_draw_particles(d_pos.data,
                                        d_vel.data,
                                        d_tag.data,
                                        m_pdata->getBox(),
                                        d_boxes.data,
                                        d_ranges.data,
                                        m_num_boxes,
                                        m_N_fill,
                                        m_mpcd_pdata->getMass(),
                                        m_type,
                                        m_first_tag,
                                        first_idx,
                                        m_T->getValue(timestep),
                                        timestep,
                                        m_seed,
                                        m_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner->end();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SlitPoreGeometryFillerGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::SlitPoreGeometryFillerGPU, std::shared_ptr<mpcd::SlitPoreGeometryFillerGPU>>
        (m, "SlitPoreGeometryFillerGPU", py::base<mpcd::SlitPoreGeometryFiller>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      Scalar,
                      unsigned int,
                      std::shared_ptr<::Variant>,
                      unsigned int,
                      std::shared_ptr<const mpcd::detail::SlitPoreGeometry>>())
        ;
    }
