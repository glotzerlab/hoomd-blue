#include "Trigger.h"

void export_Trigger(pybind11::module& m)
    {
    pybind11::class_<Trigger, std::shared_ptr<Trigger> >(m,"Trigger")
        .def("__call__", &Trigger::operator())
        ;

    pybind11::class_<PeriodicTrigger, Trigger, std::shared_ptr<PeriodicTrigger> >(m, "PeriodicTrigger")
        .def(pybind11::init< uint64_t, uint64_t >(), pybind11::arg("period"), pybind11::arg("phase"))
        .def(pybind11::init< uint64_t >(), pybind11::arg("period"))
        .def_property("phase", &PeriodicTrigger::getPhase, &PeriodicTrigger::setPhase)
        .def_property("period", &PeriodicTrigger::getPeriod, &PeriodicTrigger::setPeriod)
        ;
    }
