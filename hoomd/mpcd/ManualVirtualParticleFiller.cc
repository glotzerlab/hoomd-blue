// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ManualVirtualParticleFiller.cc
 * \brief Definition of mpcd::ManualVirtualParticleFiller
 */

#include "ManualVirtualParticleFiller.h"

namespace hoomd
    {
mpcd::ManualVirtualParticleFiller::ManualVirtualParticleFiller(
    std::shared_ptr<SystemDefinition> sysdef,
    const std::string& type,
    Scalar density,
    std::shared_ptr<Variant> T)
    : mpcd::VirtualParticleFiller(sysdef, type, density, T), m_N_fill(0), m_first_tag(0),
      m_first_idx(0)
    {
    }

void mpcd::ManualVirtualParticleFiller::fill(uint64_t timestep)
    {
    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // update the fill volume
    computeNumFill();

    // get the first tag from the fill number
    m_first_tag = computeFirstTag(m_N_fill);

    // add the new virtual particles locally
    m_first_idx = m_mpcd_pdata->addVirtualParticles(m_N_fill);

    // draw the particles consistent with those tags
    drawParticles(timestep);

    m_mpcd_pdata->invalidateCellCache();
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_ManualVirtualParticleFiller(pybind11::module& m)
    {
    pybind11::class_<mpcd::ManualVirtualParticleFiller,
                     mpcd::VirtualParticleFiller,
                     std::shared_ptr<mpcd::ManualVirtualParticleFiller>>(
        m,
        "ManualVirtualParticleFiller")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
