//
// Created by girard01 on 11/9/22.
//

#include "Thermostat.h"
namespace hoomd::md::detail{
    void export_Thermostat(pybind11::module& m){
        pybind11::class_<Thermostat, std::shared_ptr<Thermostat>>(m, "Thermostat")
        .def(pybind11::init<std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>>())
        .def_property("T", &Thermostat::getT, &Thermostat::setT)
        .def_property("translationalDOF", &Thermostat::getTranslationalDOF, &Thermostat::setTranslationalDOF)
        .def_property("rotationalDOF", &Thermostat::getRotationalDOF, &Thermostat::setRotationalDOF)
        .def("thermalizeThermostat", &Thermostat::thermalizeThermostat);
    }

    void export_MTTKThermostat(pybind11::module& m){
        pybind11::class_<MTTKThermostat, Thermostat, std::shared_ptr<MTTKThermostat>>(m, "MTTKThermostat")
        .def(pybind11::init<std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>, Scalar>())
        .def_property("tau", &MTTKThermostat::getTau, &MTTKThermostat::setTau);
    }

    void export_BussiThermostat(pybind11::module& m){
        pybind11::class_<BussiThermostat, Thermostat, std::shared_ptr<BussiThermostat>>(m, "BussiThermostat")
        .def(pybind11::init<std::shared_ptr<ParticleGroup>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>, uint16_t>());
    }
}