// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PythonTuner.h"

#include <exception>
#include <string>

namespace hoomd
    {
PythonTuner::PythonTuner(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Trigger> trigger,
                         pybind11::object tuner)
    : Tuner(sysdef, trigger)
    {
    setTuner(tuner);
    }

void PythonTuner::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_tuner.attr("act")(timestep);
    }

void PythonTuner::setTuner(pybind11::object tuner)
    {
    m_tuner = tuner;
    auto flags = PDataFlags();
    for (auto flag : tuner.attr("flags"))
        {
        flags.set(flag.cast<size_t>());
        }
    m_flags = flags;
    }

PDataFlags PythonTuner::getRequestedPDataFlags()
    {
    return m_flags;
    }

namespace detail
    {
void export_PythonTuner(pybind11::module& m)
    {
    pybind11::class_<PythonTuner, Tuner, std::shared_ptr<PythonTuner>>(m, "PythonTuner")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            pybind11::object>());
    }

    } // end namespace detail

    } // end namespace hoomd
