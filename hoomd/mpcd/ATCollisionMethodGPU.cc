// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ATCollisionMethodGPU.h
 * \brief Definition of mpcd::ATCollisionMethodGPU
 */

#include "ATCollisionMethodGPU.h"
#include "ATCollisionMethodGPU.cuh"

mpcd::ATCollisionMethodGPU::ATCollisionMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                 unsigned int cur_timestep,
                                                 unsigned int period,
                                                 int phase,
                                                 unsigned int seed,
                                                 std::shared_ptr<mpcd::CellThermoCompute> thermo,
                                                 std::shared_ptr<mpcd::CellThermoCompute> rand_thermo,
                                                 std::shared_ptr<::Variant> T)
    : mpcd::ATCollisionMethod(sysdata,cur_timestep,period,phase,seed,thermo,rand_thermo,T)
    {
    m_tuner_draw.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_at_draw", m_exec_conf));
    m_tuner_apply.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_at_apply", m_exec_conf));
    }

void mpcd::ATCollisionMethodGPU::drawVelocities(unsigned int timestep)
    {
    // mpcd particle data
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_alt_vel(m_mpcd_pdata->getAltVelocities(), access_location::device, access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // random velocities are drawn for each particle and stored into the "alternate" arrays
    const Scalar T = m_T->getValue(timestep);

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_idx(m_embed_group->getIndexArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_alt_vel_embed(m_pdata->getAltVelocities(), access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_embed(m_pdata->getTags(), access_location::device, access_mode::read);
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
                                    m_seed,
                                    T,
                                    N_mpcd,
                                    N_tot,
                                    m_tuner_draw->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
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
                                    m_seed,
                                    T,
                                    N_mpcd,
                                    N_tot,
                                    m_tuner_draw->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner_draw->end();
        }
    }

void mpcd::ATCollisionMethodGPU::applyVelocities()
    {
    // mpcd particle data
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel_alt(m_mpcd_pdata->getAltVelocities(), access_location::device, access_mode::read);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // cell data
    ArrayHandle<double4> d_cell_vel(m_thermo->getCellVelocities(), access_location::device, access_mode::read);
    ArrayHandle<double4> d_rand_vel(m_rand_thermo->getCellVelocities(), access_location::device, access_mode::read);

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_idx(m_embed_group->getIndexArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel_alt_embed(m_pdata->getAltVelocities(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_embed_cell_ids(m_cl->getEmbeddedGroupCellIds(), access_location::device, access_mode::read);
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
                                     m_tuner_apply->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
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
                                     m_tuner_apply->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner_apply->end();
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ATCollisionMethodGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::ATCollisionMethodGPU, std::shared_ptr<mpcd::ATCollisionMethodGPU> >
        (m, "ATCollisionMethodGPU", py::base<mpcd::ATCollisionMethod>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>,
                      unsigned int,
                      unsigned int,
                      int,
                      unsigned int,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<mpcd::CellThermoCompute>,
                      std::shared_ptr<::Variant>>())
    ;
    }
