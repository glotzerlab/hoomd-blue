#include "PythonUpdater.h"

#include <exception>
#include <string>

PythonUpdater::PythonUpdater(std::shared_ptr<SystemDefinition> sysdef,
                             pybind11::object updater) : Updater(sysdef)
    {
    setUpdater(updater);
    }

void PythonUpdater::update(unsigned int timestep)
    {
    m_updater.attr("act")(timestep);
    }

void PythonUpdater::setUpdater(pybind11::object updater)
    {
    m_updater = updater;
    auto flags = PDataFlags();
    for (auto flag: updater.attr("flags"))
        {
        flags.set(flag.cast<size_t>());
        }
    m_flags = flags;
    }

PDataFlags PythonUpdater::getRequestedPDataFlags()
    {
    return m_flags;
    }

void export_PythonUpdater(pybind11::module& m)
    {
    pybind11::class_<PythonUpdater, Updater, std::shared_ptr<PythonUpdater>
                    >(m, "PythonUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::object>())
        ;
    }
