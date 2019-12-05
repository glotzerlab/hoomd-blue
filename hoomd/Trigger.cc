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
        bool operator()(uint64_t timestep) override
            {
            PYBIND11_OVERLOAD_NAME(bool,         // Return type
                                   Trigger,      // Parent class
                                   "__call__",   // name of function in python
                                   operator(),   // Name of function in C++
                                   timestep      // Argument(s)
                              );
        }
    };

void export_Trigger(pybind11::module& m)
    {
    pybind11::class_<Trigger, TriggerPy, std::shared_ptr<Trigger> >(m,"Trigger")
        .def(pybind11::init<>())
        .def("__call__", &Trigger::operator())
        ;

    pybind11::class_<PeriodicTrigger, Trigger, std::shared_ptr<PeriodicTrigger> >(m, "PeriodicTrigger")
        .def(pybind11::init< uint64_t, uint64_t >(), pybind11::arg("period"), pybind11::arg("phase"))
        .def(pybind11::init< uint64_t >(), pybind11::arg("period"))
        .def_property("phase", &PeriodicTrigger::getPhase, &PeriodicTrigger::setPhase)
        .def_property("period", &PeriodicTrigger::getPeriod, &PeriodicTrigger::setPeriod)
        ;

    m.def("_test_trigger_call", &testTriggerCall);
    }
