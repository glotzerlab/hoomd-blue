// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "Trigger.h"

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

    pybind11::class_<PeriodicTrigger, Trigger, std::shared_ptr<PeriodicTrigger> >(m, "PeriodicTrigger")
        .def(pybind11::init< uint64_t, uint64_t >(), pybind11::arg("period"), pybind11::arg("phase"))
        .def(pybind11::init< uint64_t >(), pybind11::arg("period"))
        .def_property("phase", &PeriodicTrigger::getPhase, &PeriodicTrigger::setPhase)
        .def_property("period", &PeriodicTrigger::getPeriod, &PeriodicTrigger::setPeriod)
        ;

    pybind11::class_<UntilTrigger, Trigger, std::shared_ptr<UntilTrigger>
                    >(m, "UntilTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("until"))
        .def_property("until", &UntilTrigger::getUntil, &UntilTrigger::setUntil)
        ;

    pybind11::class_<AfterTrigger, Trigger, std::shared_ptr<AfterTrigger>
                    >(m, "AfterTrigger")
        .def(pybind11::init<uint64_t>(), pybind11::arg("after"))
        .def_property("after", &AfterTrigger::getAfter, &AfterTrigger::setAfter)
        ;

    pybind11::class_<NotTrigger, Trigger, std::shared_ptr<NotTrigger>
                    >(m, "NotTrigger")
        .def(pybind11::init<std::shared_ptr<Trigger> >(),
             pybind11::arg("trigger"))
        .def_property("trigger",
                      &NotTrigger::getTrigger,
                      &NotTrigger::setTrigger)
        ;

    pybind11::class_<AndTrigger, Trigger, std::shared_ptr<AndTrigger>
                    >(m, "AndTrigger")
        .def(pybind11::init<std::shared_ptr<Trigger>,
                            std::shared_ptr<Trigger> >(),
             pybind11::arg("trigger1"),
             pybind11::arg("trigger2"))
        .def_property("trigger1",
                      &AndTrigger::getTrigger1,
                      &AndTrigger::setTrigger1)
        .def_property("trigger2",
                      &AndTrigger::getTrigger2,
                      &AndTrigger::setTrigger2)
        ;

    pybind11::class_<OrTrigger, Trigger, std::shared_ptr<OrTrigger>
                    >(m, "OrTrigger")
        .def(pybind11::init<std::shared_ptr<Trigger>,
                            std::shared_ptr<Trigger> >(),
             pybind11::arg("trigger1"),
             pybind11::arg("trigger2"))
        .def_property("trigger1",
                      &OrTrigger::getTrigger1,
                      &OrTrigger::setTrigger1)
        .def_property("trigger2",
                      &OrTrigger::getTrigger2,
                      &OrTrigger::setTrigger2)
        ;

    m.def("_test_trigger_call", &testTriggerCall);
    }
