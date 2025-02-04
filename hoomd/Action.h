// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

#include "Autotuned.h"
#include "SharedSignal.h"
#include "SystemDefinition.h"

namespace hoomd
    {

/// Base class for actions that act on the simulation state
/*! Compute, Updater, Analyzer, and Tuner inherit common methods from Action.

    Some, but not all classes in HOOMD provide autotuners. To give the user a unified API to query
    and interact with these autotuners, Action provides a pybind11 interface to get and set
    autotuner parameters for all child classes. Derived classes must add all autotuners to
    m_autotuners for the base class API to be effective.
*/
class Action : public Autotuned
    {
    public:
    Action(std::shared_ptr<SystemDefinition> sysdef)
        : m_sysdef(sysdef), m_pdata(sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf())
        {
        }

    protected:
    /// The system definition this action is associated with.
    const std::shared_ptr<SystemDefinition> m_sysdef;

    /// The particle data this action is associated with.
    const std::shared_ptr<ParticleData> m_pdata;

    /// The simulation's execution configuration.
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

    /// Stored shared ptr to the system signals
    std::vector<std::shared_ptr<hoomd::detail::SignalSlot>> m_slots;

    void addSlot(std::shared_ptr<hoomd::detail::SignalSlot> slot)
        {
        m_slots.push_back(slot);
        }

    void removeDisconnectedSlots()
        {
        for (unsigned int i = 0; i < m_slots.size();)
            {
            if (!m_slots[i]->connected())
                {
                m_exec_conf->msg->notice(8) << "Found dead signal @" << std::hex << m_slots[i].get()
                                            << std::dec << std::endl;
                m_slots.erase(m_slots.begin() + i);
                }
            else
                {
                i++;
                }
            }
        }
    };

namespace detail
    {
/// Exports the Action class to python.
void export_Action(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
