// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Updater.h"

/*! \file Updater.cc
    \brief Defines a base class for all updaters
*/

namespace hoomd
    {
/*! \param sysdef System this compute will act on. Must not be NULL.
    \post The Updater is constructed with the given particle data.
*/
Updater::Updater(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
    {
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    }

namespace detail
    {
void export_Updater(pybind11::module& m)
    {
    pybind11::class_<Updater, std::shared_ptr<Updater>>(m, "Updater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("update", &Updater::update)
        .def("notifyDetach", &Updater::notifyDetach);
    }

    } // end namespace detail

    } // end namespace hoomd
