// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include <pybind11/pybind11.h>

#include "Tuner.h"

namespace hoomd
    {
class PYBIND11_EXPORT PythonTuner : public Tuner
    {
    public:
    PythonTuner(std::shared_ptr<SystemDefinition> sysdef,
                std::shared_ptr<Trigger> trigger,
                pybind11::object tuner);

    void update(uint64_t timestep);

    PDataFlags getRequestedPDataFlags();

    void setTuner(pybind11::object tuner);

    pybind11::object getTuner()
        {
        return m_tuner;
        }

    protected:
    pybind11::object m_tuner;
    PDataFlags m_flags;
    };

namespace detail
    {
void export_PythonTuner(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
