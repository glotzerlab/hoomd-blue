// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFillerGPU.cc
 * \brief Definition of mpcd::ParallelPlateGeometryFillerGPU
 */

#include "ParallelPlateGeometryFillerGPU.cuh"
#include "ParallelPlateGeometryFillerGPU.h"

namespace hoomd
    {
mpcd::ParallelPlateGeometryFillerGPU::ParallelPlateGeometryFillerGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    const std::string& type,
    Scalar density,
    std::shared_ptr<Variant> T,
    std::shared_ptr<const mpcd::ParallelPlateGeometry> geom)
    : mpcd::ParallelPlateGeometryFiller(sysdef, type, density, T, geom)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "mpcd_slit_filler"));
    m_autotuners.push_back(m_tuner);
    }

/*!
 * \param timestep Current timestep
 */
void mpcd::ParallelPlateGeometryFillerGPU::drawParticles(uint64_t timestep)
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

    uint16_t seed = m_sysdef->getSeed();

    m_tuner->begin();
    mpcd::gpu::slit_draw_particles(d_pos.data,
                                   d_vel.data,
                                   d_tag.data,
                                   *m_geom,
                                   m_y_min,
                                   m_y_max,
                                   m_pdata->getBox(),
                                   m_mpcd_pdata->getMass(),
                                   m_type,
                                   m_N_lo,
                                   m_N_hi,
                                   m_first_tag,
                                   m_first_idx,
                                   (*m_T)(timestep),
                                   timestep,
                                   seed,
                                   m_filler_id,
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
void export_ParallelPlateGeometryFillerGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::ParallelPlateGeometryFillerGPU,
                     mpcd::ParallelPlateGeometryFiller,
                     std::shared_ptr<mpcd::ParallelPlateGeometryFillerGPU>>(
        m,
        "ParallelPlateGeometryFillerGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const mpcd::ParallelPlateGeometry>>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
