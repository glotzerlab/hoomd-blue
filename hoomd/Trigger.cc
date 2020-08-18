// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "Trigger.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<Trigger> >);

//* Method to enable unit testing of C++ trigger calls from pytest
bool testTriggerCall(std::shared_ptr<Trigger> t, uint64_t step)
    {
    return (*t)(step);
    }

//* Trampoline for classes inherited in python
class TriggerPy : public Trigger
    {
    public:
        // Inherit the constructors
        using Trigger::Trigger;

        // trampoline method
        bool compute(uint64_t timestep) override
            {
            PYBIND11_OVERLOAD_PURE(bool,         // Return type
                                   Trigger,      // Parent class
                                   compute,
                                   timestep      // Argument(s)
                              );
            }
    };

void export_Trigger(pybind11::module& m)
    {
    pybind11::class_<Trigger, TriggerPy, std::shared_ptr<Trigger> >(m,"Trigger")
        .def(pybind11::init<>())
        .def("__call__", &Trigger::operator())
        .def("compute", &Trigger::compute)
        ;

    pybind11::class_<PeriodicTrigger, Trigger,
                     std::shared_ptr<PeriodicTrigger> >(m, "PeriodicTrigger")
        .def(pybind11::init< uint64_t, uint64_t >(),
             pybind11::arg("period"),
             pybind11::arg("phase"))
        .def(pybind11::init< uint64_t >(), pybind11::arg("period"))
        .def_property("phase",
                      &PeriodicTrigger::getPhase,
                      &PeriodicTrigger::setPhase)
        .def_property("period",
                      &PeriodicTrigger::getPeriod,
                      &PeriodicTrigger::setPeriod)
        ;

    pybind11::class_<BeforeTrigger, Trigger, std::shared_ptr<BeforeTrigger>
                    >(m, "BeforeTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property("timestep",
                      &BeforeTrigger::getTimestep,
                      &BeforeTrigger::setTimestep)
        ;

    pybind11::class_<OnTrigger, Trigger, std::shared_ptr<OnTrigger>
                    >(m, "OnTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property("timestep",
                      &OnTrigger::getTimestep,
                      &OnTrigger::setTimestep)
        ;

    pybind11::class_<AfterTrigger, Trigger, std::shared_ptr<AfterTrigger>
                    >(m, "AfterTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("timestep"))
        .def_property("timestep",
                      &AfterTrigger::getTimestep,
                      &AfterTrigger::setTimestep)
        ;

    pybind11::class_<NotTrigger, Trigger, std::shared_ptr<NotTrigger>
                    >(m, "NotTrigger")
        .def(pybind11::init<std::shared_ptr<Trigger> >(),
             pybind11::arg("trigger"))
        .def_property("trigger",
                      &NotTrigger::getTrigger,
                      &NotTrigger::setTrigger)
        ;

    pybind11::bind_vector<std::vector<std::shared_ptr<Trigger> >
                         >(m, "trigger_list");

    pybind11::class_<AndTrigger, Trigger, std::shared_ptr<AndTrigger>
                    >(m, "AndTrigger")
        .def(pybind11::init<pybind11::object>(),
             pybind11::arg("triggers"))
        .def_property_readonly("triggers", &AndTrigger::getTriggers)
        ;

    pybind11::class_<OrTrigger, Trigger, std::shared_ptr<OrTrigger>
                    >(m, "OrTrigger")
        .def(pybind11::init<pybind11::object>(),
             pybind11::arg("triggers"))
        .def_property_readonly("triggers", &OrTrigger::getTriggers)
        ;

    m.def("_test_trigger_call", &testTriggerCall);
    }
