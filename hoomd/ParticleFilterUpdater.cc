// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ParticleFilterUpdater.h"
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::ParticleGroup>>);

namespace hoomd
    {
ParticleFilterUpdater::ParticleFilterUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Trigger> trigger,
                                             std::vector<std::shared_ptr<ParticleGroup>> groups)
    : Updater(sysdef, trigger), m_groups(groups)
    {
    }

ParticleFilterUpdater::~ParticleFilterUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying ParticleFilterUpdater\n";
    }

/// Update filters
void ParticleFilterUpdater::update(uint64_t timestep)
    {
    for (auto& group : m_groups)
        {
        group->updateMemberTags(true);
        }
    }

namespace detail
    {
/// Export the BoxResizeUpdater to python
void export_ParticleFilterUpdater(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<ParticleGroup>>>(m, "ParticleGroupList");
    pybind11::class_<ParticleFilterUpdater, Updater, std::shared_ptr<ParticleFilterUpdater>>(
        m,
        "ParticleFilterUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def_property_readonly("groups", &ParticleFilterUpdater::getGroups);
    }

    } // end namespace detail

    } // end namespace hoomd
