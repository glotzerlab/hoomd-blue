// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/VirtualParticleFiller.cc
 * \brief Definition of mpcd::VirtualParticleFiller
 */

#include "VirtualParticleFiller.h"

namespace hoomd
    {
unsigned int mpcd::VirtualParticleFiller::s_filler_count = 0;

mpcd::VirtualParticleFiller::VirtualParticleFiller(std::shared_ptr<SystemDefinition> sysdef,
                                                   const std::string& type,
                                                   Scalar density,
                                                   std::shared_ptr<Variant> T)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_density(density), m_T(T)
    {
    setType(type);

    // assign ID from count, but synchronize with root in case count got off somewhere
    m_filler_id = s_filler_count++;
#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        bcast(m_filler_id, 0, m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI
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

std::string mpcd::VirtualParticleFiller::getType() const
    {
    return m_mpcd_pdata->getNameByType(m_type);
    }

void mpcd::VirtualParticleFiller::setType(const std::string& type)
    {
    m_type = m_mpcd_pdata->getTypeByName(type);
    }

/*!
 * \param N_fill Number of virtual particles to add on each rank
 * \returns First tag to assign to a virtual particle on each rank.
 */
unsigned int mpcd::VirtualParticleFiller::computeFirstTag(unsigned int N_fill) const
    {
    // exclusive scan of number on each rank in MPI
    unsigned int first_tag = 0;
#ifdef ENABLE_MPI
    if (m_exec_conf->getNRanks() > 1)
        {
        MPI_Exscan(&N_fill,
                   &first_tag,
                   1,
                   MPI_UNSIGNED,
                   MPI_SUM,
                   m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI

    // shift the first tag based on the number of particles (real and virtual) already used
    first_tag += m_mpcd_pdata->getNGlobal() + m_mpcd_pdata->getNVirtualGlobal();

    return first_tag;
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_VirtualParticleFiller(pybind11::module& m)
    {
    pybind11::class_<mpcd::VirtualParticleFiller, std::shared_ptr<mpcd::VirtualParticleFiller>>(
        m,
        "VirtualParticleFiller")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>>())
        .def_property("density",
                      &mpcd::VirtualParticleFiller::getDensity,
                      &mpcd::VirtualParticleFiller::setDensity)
        .def_property("type",
                      &mpcd::VirtualParticleFiller::getType,
                      &mpcd::VirtualParticleFiller::setType)
        .def_property("kT",
                      &mpcd::VirtualParticleFiller::getTemperature,
                      &mpcd::VirtualParticleFiller::setTemperature);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
