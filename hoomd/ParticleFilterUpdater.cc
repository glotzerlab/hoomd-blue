#include "ParticleFilterUpdater.h"
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<ParticleGroup>>);

ParticleFilterUpdater::ParticleFilterUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                             std::vector<std::shared_ptr<ParticleGroup>> groups)
    : Updater(sysdef), m_groups(groups)
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

/// Export the BoxResizeUpdater to python
void export_ParticleFilterUpdater(pybind11::module& m)
    {
    py::bind_vector<std::vector<std::shared_ptr<ParticleGroup>>>(m, "ParticleGroupList");
    py::class_<ParticleFilterUpdater, Updater, std::shared_ptr<ParticleFilterUpdater>>(
        m,
        "ParticleFilterUpdater")
        .def(py::init<std::shared_ptr<SystemDefinition>>())
        .def_property_readonly("groups", &ParticleFilterUpdater::getGroups);
    }
