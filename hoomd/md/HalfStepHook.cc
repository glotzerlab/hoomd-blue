// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HalfStepHook.h"

namespace hoomd
    {
namespace md
    {

//! Trampoline for HalfStepHook classes inherited in python
class PYBIND11_EXPORT HalfStepHookPy : public HalfStepHook
    {
    public:
    // Inherit the constructor
    using HalfStepHook::HalfStepHook;

    // Trampoline methods
    void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) override
        {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, setSystemDefinition, sysdef);
        }

    void update(uint64_t timestep) override
        {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, update, timestep);
        }
    };

namespace detail
    {

// Method to enable unit testing of C++ HalfStepHook::update from pytest
void testHalfStepHookUpdate(std::shared_ptr<HalfStepHook> hook, uint64_t step)
    {
    return hook->update(step);
    }

void export_HalfStepHook(pybind11::module& m)
    {
    pybind11::class_<HalfStepHook, HalfStepHookPy, std::shared_ptr<HalfStepHook>>(m, "HalfStepHook")
        .def(pybind11::init_alias<>())
        .def("setSystemDefinition", &HalfStepHook::setSystemDefinition)
        .def("update", &HalfStepHook::update);
    }

    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
