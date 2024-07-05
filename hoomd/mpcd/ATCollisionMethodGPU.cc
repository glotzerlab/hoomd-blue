// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ATCollisionMethodGPU.h
 * \brief Definition of mpcd::ATCollisionMethodGPU
 */

#include "ATCollisionMethodGPU.h"
#include "ATCollisionMethodGPU.cuh"
#include "CellThermoComputeGPU.h"

namespace hoomd
    {
mpcd::ATCollisionMethodGPU::ATCollisionMethodGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                 uint64_t cur_timestep,
                                                 uint64_t period,
                                                 int phase,
                                                 std::shared_ptr<Variant> T)
    : mpcd::ATCollisionMethod(sysdef, cur_timestep, period, phase, T)
    {
    m_tuner_draw.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_at_draw"));
    m_tuner_apply.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "mpcd_at_apply"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_draw, m_tuner_apply});
    }

void mpcd::ATCollisionMethodGPU::drawVelocities(uint64_t timestep)
    {
    // mpcd particle data
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar4> d_alt_vel(m_mpcd_pdata->getAltVelocities(),
                                   access_location::device,
                                   access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // random velocities are drawn for each particle and stored into the "alternate" arrays
    const Scalar T = (*m_T)(timestep);

    uint16_t seed = m_sysdef->getSeed();

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_idx(m_embed_group->getIndexArray(),
                                              access_location::device,
                                              access_mode::read);
        ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(),
                                         access_location::device,
                                         access_mode::read);
        ArrayHandle<Scalar4> d_alt_vel_embed(m_pdata->getAltVelocities(),
                                             access_location::device,
                                             access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_embed(m_pdata->getTags(),
                                              access_location::device,
                                              access_mode::read);
        N_tot += m_embed_group->getNumMembers();

        m_tuner_draw->begin();
        mpcd::gpu::at_draw_velocity(d_alt_vel.data,
                                    d_alt_vel_embed.data,
                                    d_tag.data,
                                    m_mpcd_pdata->getMass(),
                                    d_embed_idx.data,
                                    d_vel_embed.data,
                                    d_tag_embed.data,
                                    timestep,
                                    seed,
                                    T,
                                    N_mpcd,
                                    N_tot,
                                    m_tuner_draw->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_draw->end();
        }
    else
        {
        m_tuner_draw->begin();
        mpcd::gpu::at_draw_velocity(d_alt_vel.data,
                                    NULL,
                                    d_tag.data,
                                    m_mpcd_pdata->getMass(),
                                    NULL,
                                    NULL,
                                    NULL,
                                    timestep,
                                    seed,
                                    T,
                                    N_mpcd,
                                    N_tot,
                                    m_tuner_draw->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_draw->end();
        }
    }

void mpcd::ATCollisionMethodGPU::applyVelocities()
    {
    // mpcd particle data
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel_alt(m_mpcd_pdata->getAltVelocities(),
                                   access_location::device,
                                   access_mode::read);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // cell data
    ArrayHandle<double4> d_cell_vel(m_thermo->getCellVelocities(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<double4> d_rand_vel(m_rand_thermo->getCellVelocities(),
                                    access_location::device,
                                    access_mode::read);

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_idx(m_embed_group->getIndexArray(),
                                              access_location::device,
                                              access_mode::read);
        ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel_alt_embed(m_pdata->getAltVelocities(),
                                             access_location::device,
                                             access_mode::read);
        ArrayHandle<unsigned int> d_embed_cell_ids(m_cl->getEmbeddedGroupCellIds(),
                                                   access_location::device,
                                                   access_mode::read);
        N_tot += m_embed_group->getNumMembers();

        m_tuner_apply->begin();
        mpcd::gpu::at_apply_velocity(d_vel.data,
                                     d_vel_embed.data,
                                     d_vel_alt.data,
                                     d_embed_idx.data,
                                     d_vel_alt_embed.data,
                                     d_embed_cell_ids.data,
                                     d_cell_vel.data,
                                     d_rand_vel.data,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_apply->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_apply->end();
        }
    else
        {
        m_tuner_apply->begin();
        mpcd::gpu::at_apply_velocity(d_vel.data,
                                     NULL,
                                     d_vel_alt.data,
                                     NULL,
                                     NULL,
                                     NULL,
                                     d_cell_vel.data,
                                     d_rand_vel.data,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_apply->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_apply->end();
        }
    }

void mpcd::ATCollisionMethodGPU::setCellList(std::shared_ptr<mpcd::CellList> cl)
    {
    if (cl != m_cl)
        {
        CollisionMethod::setCellList(cl);

        detachCallbacks();
        if (m_cl)
            {
            m_thermo = std::make_shared<mpcd::CellThermoComputeGPU>(m_sysdef, m_cl);
            m_rand_thermo = std::make_shared<mpcd::CellThermoComputeGPU>(m_sysdef, m_cl);
            attachCallbacks();
            }
        else
            {
            m_thermo = std::shared_ptr<mpcd::CellThermoComputeGPU>();
            m_rand_thermo = std::shared_ptr<mpcd::CellThermoComputeGPU>();
            }
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ATCollisionMethodGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::ATCollisionMethodGPU,
                     mpcd::ATCollisionMethod,
                     std::shared_ptr<mpcd::ATCollisionMethodGPU>>(m, "ATCollisionMethodGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            uint64_t,
                            uint64_t,
                            int,
                            std::shared_ptr<Variant>>());
    }

    } // end namespace hoomd
