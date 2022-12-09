// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.


#include "Thermostat.h"
namespace hoomd::md::detail{
    void export_Thermostat(pybind11::module& m){
        pybind11::class_<Thermostat, std::shared_ptr<Thermostat>>(m, "Thermostat")
        .def(pybind11::init<std::shared_ptr<Variant>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<SystemDefinition>>())
        .def_property("kT", &Thermostat::getT, &Thermostat::setT)
        .def("getThermostatEnergy", &Thermostat::getThermostatEnergy)
        .def("thermalizeThermostat", &Thermostat::thermalizeThermostat);
    }

    void export_MTTKThermostat(pybind11::module& m){
        pybind11::class_<MTTKThermostat, Thermostat, std::shared_ptr<MTTKThermostat>>(m, "MTTKThermostat")
        .def(pybind11::init<std::shared_ptr<Variant>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("translationalDOF", &MTTKThermostat::getTranslationalDOF, &MTTKThermostat::setTranslationalDOF)
        .def_property("rotationalDOF", &MTTKThermostat::getRotationalDOF, &MTTKThermostat::setRotationalDOF)
        .def_property("tau", &MTTKThermostat::getTau, &MTTKThermostat::setTau);

    }

    void export_BussiThermostat(pybind11::module& m){
        pybind11::class_<BussiThermostat, Thermostat, std::shared_ptr<BussiThermostat>>(m, "BussiThermostat")
        .def(pybind11::init< std::shared_ptr<Variant>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<SystemDefinition>>());
    }

    void export_BerendsenThermostat(pybind11::module& m){
        pybind11::class_<BerendsenThermostat, Thermostat, std::shared_ptr<BerendsenThermostat>>(m, "BerendsenThermostat")
        .def(pybind11::init<std::shared_ptr<Variant>, std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<SystemDefinition>, Scalar>());
    }
}
