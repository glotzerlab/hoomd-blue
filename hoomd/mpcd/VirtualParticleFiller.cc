// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/VirtualParticleFiller.cc
 * \brief Definition of mpcd::VirtualParticleFiller
 */

#include "VirtualParticleFiller.h"

namespace hoomd
    {
mpcd::VirtualParticleFiller::VirtualParticleFiller(std::shared_ptr<SystemDefinition> sysdef,
                                                   Scalar density,
                                                   unsigned int type,
                                                   std::shared_ptr<Variant> T)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_density(density), m_type(type), m_T(T),
      m_N_fill(0), m_first_tag(0)
    {
    }

void mpcd::VirtualParticleFiller::fill(uint64_t timestep)
    {
    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // update the fill volume
    computeNumFill();

    // in mpi, do a prefix scan on the tag offset in this range
    // then shift the first tag by the current number of particles, which ensures a compact tag
    // array
    m_first_tag = 0;
#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        // scan the number to fill to get the tag range I own
        MPI_Exscan(&m_N_fill,
                   &m_first_tag,
                   1,
                   MPI_UNSIGNED,
                   MPI_SUM,
                   m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI
    m_first_tag += m_mpcd_pdata->getNGlobal() + m_mpcd_pdata->getNVirtualGlobal();

    // add the new virtual particles locally
    m_mpcd_pdata->addVirtualParticles(m_N_fill);

    // draw the particles consistent with those tags
    drawParticles(timestep);

    m_mpcd_pdata->invalidateCellCache();
    }

void mpcd::VirtualParticleFiller::setDensity(Scalar density)
    {
    if (density <= Scalar(0.0))
        {
        m_exec_conf->msg->error() << "MPCD virtual particle density must be positive" << std::endl;
        throw std::runtime_error("Invalid virtual particle density");
        }
    m_density = density;
    }

void mpcd::VirtualParticleFiller::setType(unsigned int type)
    {
    if (type >= m_mpcd_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Invalid type id specified for MPCD virtual particle filler"
                                  << std::endl;
        throw std::runtime_error("Invalid type id");
        }
    m_type = type;
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_VirtualParticleFiller(pybind11::module& m)
    {
    pybind11::class_<mpcd::VirtualParticleFiller, std::shared_ptr<mpcd::VirtualParticleFiller>>(
        m,
        "VirtualParticleFiller")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            Scalar,
                            unsigned int,
                            std::shared_ptr<Variant>>())
        .def("setDensity", &mpcd::VirtualParticleFiller::setDensity)
        .def("setType", &mpcd::VirtualParticleFiller::setType)
        .def("setTemperature", &mpcd::VirtualParticleFiller::setTemperature);
    }

    } // end namespace hoomd
