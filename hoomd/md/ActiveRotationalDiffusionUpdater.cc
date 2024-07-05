// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ActiveRotationalDiffusionUpdater.cc
    \brief Defines the ActiveRotationalDiffusionUpdater class
*/

#include "ActiveRotationalDiffusionUpdater.h"

#include <iostream>

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System definition
 *  \param rotational_diffusion The diffusion across time
 *  \param group the particles to diffusion rotation on
 */

ActiveRotationalDiffusionUpdater::ActiveRotationalDiffusionUpdater(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    std::shared_ptr<Variant> rotational_diffusion,
    std::shared_ptr<ActiveForceCompute> active_force)
    : Updater(sysdef, trigger), m_rotational_diffusion(rotational_diffusion),
      m_active_force(active_force)
    {
    assert(m_pdata);
    assert(m_rotational_diffusion);
    assert(m_active_force);
    m_exec_conf->msg->notice(5) << "Constructing ActiveRotationalDiffusionUpdater" << endl;
    }

ActiveRotationalDiffusionUpdater::~ActiveRotationalDiffusionUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveRotationalDiffusionUpdater" << endl;
    }

/** Perform the needed calculations to update particle orientations
    \param timestep Current time step of the simulation
*/
void ActiveRotationalDiffusionUpdater::update(uint64_t timestep)
    {
    m_active_force->rotationalDiffusion(m_rotational_diffusion->operator()(timestep), timestep);
    }

namespace detail
    {
void export_ActiveRotationalDiffusionUpdater(pybind11::module& m)
    {
    pybind11::class_<ActiveRotationalDiffusionUpdater,
                     Updater,
                     std::shared_ptr<ActiveRotationalDiffusionUpdater>>(
        m,
        "ActiveRotationalDiffusionUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ActiveForceCompute>>())
        .def_property("rotational_diffusion",
                      &ActiveRotationalDiffusionUpdater::getRotationalDiffusion,
                      &ActiveRotationalDiffusionUpdater::setRotationalDiffusion);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
