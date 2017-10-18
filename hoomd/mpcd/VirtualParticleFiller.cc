// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/VirtualParticleFiller.cc
 * \brief Definition of mpcd::VirtualParticleFiller
 */

#include "VirtualParticleFiller.h"

mpcd::VirtualParticleFiller::VirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                                                   Scalar density,
                                                   unsigned int type,
                                                   std::shared_ptr<::Variant> T,
                                                   unsigned int seed)
    : m_sysdef(sysdata->getSystemDefinition()),
      m_pdata(m_sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_pdata(sysdata->getParticleData()),
      m_cl(sysdata->getCellList()),
      m_density(density), m_type(type), m_T(T), m_seed(seed), m_N_fill(0), m_first_tag(0), m_N_fill_global(0)
    {
    #ifdef ENABLE_MPI
    // synchronize seed from root across all ranks in MPI in case users has seeded from system time or entropy
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Bcast(&m_seed, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI
    }

void mpcd::VirtualParticleFiller::fill(unsigned int timestep)
    {
    // update the fill volume
    computeNumFill();

    // the first tag on the lowest rank is the current number of particles
    m_first_tag = m_mpcd_pdata->getNGlobal() + m_mpcd_pdata->getNVirtualGlobal();

    // in mpi, do a prefix scan on the tag offset in this range
    m_N_fill_global = m_N_fill;
    #ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        // scan the number to fill to get the tag range I own
        MPI_Exscan(&m_N_fill, &m_first_tag, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());

        // the last rank determines the total number of virtual particles and broadcasts the result
        m_N_fill_global = m_first_tag + m_N_fill;
        MPI_Bcast(&m_N_fill_global, 1, MPI_UNSIGNED, m_exec_conf->getNRanks() - 1, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    // add the new virtual particles locally
    m_mpcd_pdata->addVirtualParticles(m_N_fill);

    // draw the particles consistent with those tags
    drawParticles(timestep);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_VirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::VirtualParticleFiller, std::shared_ptr<mpcd::VirtualParticleFiller> >(m, "VirtualParticleFiller")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, Scalar, unsigned int, std::shared_ptr<::Variant>, unsigned int>());
    }
