// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SlitPoreGeometryFillerGPU.cc
 * \brief Definition of mpcd::SlitPoreGeometryFillerGPU
 */

#include "SlitPoreGeometryFillerGPU.h"
#include "SlitPoreGeometryFillerGPU.cuh"

namespace hoomd
    {
mpcd::SlitPoreGeometryFillerGPU::SlitPoreGeometryFillerGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    Scalar density,
    unsigned int type,
    std::shared_ptr<Variant> T,
    uint16_t seed,
    std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom)
    : mpcd::SlitPoreGeometryFiller(sysdef, density, type, T, seed, geom)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "mpcd_slit_filler"));
    m_autotuners.push_back(m_tuner);
    }

/*!
 * \param timestep Current timestep
 */
void mpcd::SlitPoreGeometryFillerGPU::drawParticles(uint64_t timestep)
    {
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(),
                                    access_location::device,
                                    access_mode::readwrite);

    // boxes for filling
    ArrayHandle<Scalar4> d_boxes(m_boxes, access_location::device, access_mode::read);
    ArrayHandle<uint2> d_ranges(m_ranges, access_location::device, access_mode::read);

    const unsigned int first_idx = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual() - m_N_fill;

    uint16_t seed = m_sysdef->getSeed();

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
                                        (*m_T)(timestep),
                                        timestep,
                                        seed,
                                        m_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SlitPoreGeometryFillerGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::SlitPoreGeometryFillerGPU,
                     mpcd::SlitPoreGeometryFiller,
                     std::shared_ptr<mpcd::SlitPoreGeometryFillerGPU>>(m,
                                                                       "SlitPoreGeometryFillerGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            Scalar,
                            unsigned int,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            std::shared_ptr<const mpcd::detail::SlitPoreGeometry>>());
    }

    } // end namespace hoomd
