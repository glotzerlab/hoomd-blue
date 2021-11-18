// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SRDCollisionMethod.h
 * \brief Definition of mpcd::SRDCollisionMethod
 */

#include "SRDCollisionMethodGPU.h"
#include "SRDCollisionMethodGPU.cuh"

namespace hoomd
    {
mpcd::SRDCollisionMethodGPU::SRDCollisionMethodGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                   unsigned int cur_timestep,
                                                   unsigned int period,
                                                   int phase,
                                                   uint16_t seed,
                                                   std::shared_ptr<mpcd::CellThermoCompute> thermo)
    : mpcd::SRDCollisionMethod(sysdata, cur_timestep, period, phase, seed, thermo)
    {
    m_tuner_rotvec.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_srd_vec", m_exec_conf));
    m_tuner_rotate.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_srd_rotate", m_exec_conf));
    }

void mpcd::SRDCollisionMethodGPU::drawRotationVectors(uint64_t timestep)
    {
    ArrayHandle<double3> d_rotvec(m_rotvec, access_location::device, access_mode::overwrite);

    uint16_t seed = m_sysdef->getSeed();

    if (m_T)
        {
        ArrayHandle<double> d_factors(m_factors, access_location::device, access_mode::overwrite);
        ArrayHandle<double3> d_cell_energy(m_thermo->getCellEnergies(),
                                           access_location::device,
                                           access_mode::read);

        m_tuner_rotvec->begin();
        mpcd::gpu::srd_draw_vectors(d_rotvec.data,
                                    d_factors.data,
                                    d_cell_energy.data,
                                    m_cl->getCellIndexer(),
                                    m_cl->getOriginIndex(),
                                    m_cl->getGlobalDim(),
                                    m_cl->getGlobalCellIndexer(),
                                    timestep,
                                    seed,
                                    (*m_T)(timestep),
                                    m_sysdef->getNDimensions(),
                                    m_tuner_rotvec->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_rotvec->end();
        }
    else
        {
        m_tuner_rotvec->begin();
        mpcd::gpu::srd_draw_vectors(d_rotvec.data,
                                    NULL,
                                    NULL,
                                    m_cl->getCellIndexer(),
                                    m_cl->getOriginIndex(),
                                    m_cl->getGlobalDim(),
                                    m_cl->getGlobalCellIndexer(),
                                    timestep,
                                    seed,
                                    1.0,
                                    m_sysdef->getNDimensions(),
                                    m_tuner_rotvec->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_rotvec->end();
        }
    }

void mpcd::SRDCollisionMethodGPU::rotate(uint64_t timestep)
    {
    // acquire MPCD particle data
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // acquire cell velocities and rotation vectors
    ArrayHandle<double4> d_cell_vel(m_thermo->getCellVelocities(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<double3> d_rotvec(m_rotvec, access_location::device, access_mode::read);

    // load scale factors if required
    std::unique_ptr<ArrayHandle<double>> d_factors;
    if (m_T)
        {
        d_factors.reset(
            new ArrayHandle<double>(m_factors, access_location::device, access_mode::read));
        }

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_group(m_embed_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);
        ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> d_embed_cell_ids(m_cl->getEmbeddedGroupCellIds(),
                                                   access_location::device,
                                                   access_mode::read);

        N_tot += m_embed_group->getNumMembers();

        m_tuner_rotate->begin();
        mpcd::gpu::srd_rotate(d_vel.data,
                              d_vel_embed.data,
                              d_embed_group.data,
                              d_embed_cell_ids.data,
                              d_cell_vel.data,
                              d_rotvec.data,
                              m_angle,
                              (m_T) ? d_factors->data : NULL,
                              N_mpcd,
                              N_tot,
                              m_tuner_rotate->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_rotate->end();
        }
    else
        {
        m_tuner_rotate->begin();
        mpcd::gpu::srd_rotate(d_vel.data,
                              NULL,
                              NULL,
                              NULL,
                              d_cell_vel.data,
                              d_rotvec.data,
                              m_angle,
                              (m_T) ? d_factors->data : NULL,
                              N_mpcd,
                              N_tot,
                              m_tuner_rotate->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_rotate->end();
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_SRDCollisionMethodGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::SRDCollisionMethodGPU,
                     mpcd::SRDCollisionMethod,
                     std::shared_ptr<mpcd::SRDCollisionMethodGPU>>(m, "SRDCollisionMethodGPU")
        .def(pybind11::init<std::shared_ptr<mpcd::SystemData>,
                            unsigned int,
                            unsigned int,
                            int,
                            unsigned int,
                            std::shared_ptr<mpcd::CellThermoCompute>>());
    }

    } // end namespace hoomd
