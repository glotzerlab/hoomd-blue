// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Trigger.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace hoomd
    {
//* Trampoline for classes inherited in python
class TriggerPy : public Trigger
    {
    public:
    // Inherit the constructors
    using Trigger::Trigger;

    // trampoline method
    bool compute(uint64_t timestep) override
        {
        PYBIND11_OVERLOAD_PURE(bool,    // Return type
                               Trigger, // Parent class
                               compute,
                               timestep // Argument(s)
        );
        }
    };

namespace detail
    {
//* Method to enable unit testing of C++ trigger calls from pytest
bool testTriggerCall(std::shared_ptr<Trigger> t, uint64_t step)
    {
    return (*t)(step);
    }

void export_Trigger(pybind11::module& m)
    {
    pybind11::class_<Trigger, TriggerPy, std::shared_ptr<Trigger>>(m, "Trigger")
        .def(pybind11::init<>())
        .def("__call__", &Trigger::operator())
        .def("compute", &Trigger::compute);

    pybind11::class_<PeriodicTrigger, Trigger, std::shared_ptr<PeriodicTrigger>>(m,
                                                                                 "PeriodicTrigger")
        .def(pybind11::init<uint64_t, uint64_t>(), pybind11::arg("period"), pybind11::arg("phase"))
        .def(pybind11::init<uint64_t>(), pybind11::arg("period"))
        .def_property_readonly("phase", &PeriodicTrigger::getPhase)
        .def_property_readonly("period", &PeriodicTrigger::getPeriod)
        .def(pybind11::pickle(
            [](const PeriodicTrigger& trigger)
            { return pybind11::make_tuple(trigger.getPeriod(), trigger.getPhase()); },
            [](pybind11::tuple params)
            { return PeriodicTrigger(params[0].cast<uint64_t>(), params[1].cast<uint64_t>()); }));

    pybind11::class_<BeforeTrigger, Trigger, std::shared_ptr<BeforeTrigger>>(m, "BeforeTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property_readonly("timestep", &BeforeTrigger::getTimestep)
        .def(pybind11::pickle([](const BeforeTrigger& trigger)
                              { return pybind11::make_tuple(trigger.getTimestep()); },
                              [](pybind11::tuple params)
                              { return BeforeTrigger(params[0].cast<uint64_t>()); }));

    pybind11::class_<OnTrigger, Trigger, std::shared_ptr<OnTrigger>>(m, "OnTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property_readonly("timestep", &OnTrigger::getTimestep)
        .def(pybind11::pickle(
            [](const OnTrigger& trigger) { return pybind11::make_tuple(trigger.getTimestep()); },
            [](pybind11::tuple params) { return OnTrigger(params[0].cast<uint64_t>()); }));

    pybind11::class_<AfterTrigger, Trigger, std::shared_ptr<AfterTrigger>>(m, "AfterTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property_readonly("timestep", &AfterTrigger::getTimestep)
        .def(pybind11::pickle(
            [](const AfterTrigger& trigger) { return pybind11::make_tuple(trigger.getTimestep()); },
            [](pybind11::tuple params) { return AfterTrigger(params[0].cast<uint64_t>()); }));

    pybind11::class_<NotTrigger, Trigger, std::shared_ptr<NotTrigger>>(m, "NotTrigger")
        .def(pybind11::init<std::shared_ptr<Trigger>>(), pybind11::arg("trigger"))
        .def_property_readonly("trigger", &NotTrigger::getTrigger);

    pybind11::class_<AndTrigger, Trigger, std::shared_ptr<AndTrigger>>(m, "AndTrigger")
        .def(pybind11::init<pybind11::object>(), pybind11::arg("triggers"));

    pybind11::class_<OrTrigger, Trigger, std::shared_ptr<OrTrigger>>(m, "OrTrigger")
        .def(pybind11::init<pybind11::object>(), pybind11::arg("triggers"));

    m.def("_test_trigger_call", &testTriggerCall);
    }

    } // end namespace detail

    } // end namespace hoomd
