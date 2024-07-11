// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/PlanarPoreGeometryFillerGPU.cc
 * \brief Definition of mpcd::PlanarPoreGeometryFillerGPU
 */

#include "PlanarPoreGeometryFillerGPU.h"
#include "PlanarPoreGeometryFillerGPU.cuh"

namespace hoomd
    {
mpcd::PlanarPoreGeometryFillerGPU::PlanarPoreGeometryFillerGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    const std::string& type,
    Scalar density,
    std::shared_ptr<Variant> T,
    std::shared_ptr<const mpcd::PlanarPoreGeometry> geom)
    : mpcd::PlanarPoreGeometryFiller(sysdef, type, density, T, geom)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "mpcd_slit_filler"));
    m_autotuners.push_back(m_tuner);
    }

/*!
 * \param timestep Current timestep
 */
void mpcd::PlanarPoreGeometryFillerGPU::drawParticles(uint64_t timestep)
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
                                        m_first_idx,
                                        (*m_T)(timestep),
                                        timestep,
                                        seed,
                                        m_tuner->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_PlanarPoreGeometryFillerGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::PlanarPoreGeometryFillerGPU,
                     mpcd::PlanarPoreGeometryFiller,
                     std::shared_ptr<mpcd::PlanarPoreGeometryFillerGPU>>(
        m,
        "PlanarPoreGeometryFillerGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const mpcd::PlanarPoreGeometry>>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
