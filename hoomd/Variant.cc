// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "Variant.h"

//* Method to enable unit testing of C++ trigger calls from pytest
Scalar testVariantCall(std::shared_ptr<Variant> t, uint64_t step)
    {
    return (*t)(step);
    }

//* Trampoline for classes inherited in python
class VariantPy : public Variant
    {
    public:
        // Inherit the constructors
        using Variant::Variant;

        // trampoline method
        Scalar operator()(uint64_t timestep) override
            {
            PYBIND11_OVERLOAD_NAME(Scalar,       // Return type
                                   Variant,      // Parent class
                                   "__call__",   // name of function in python
                                   operator(),   // Name of function in C++
                                   timestep      // Argument(s)
                              );
            }

        Scalar min() override
            {
            PYBIND11_OVERLOAD_PURE(Scalar,       // Return type
                                   Variant,      // Parent class
                                   min           // name of function
                    );
            }

        Scalar max() override
            {
            PYBIND11_OVERLOAD_PURE(Scalar,       // Return type
                                   Variant,      // Parent class
                                   max           // name of function
                    );
            }
    };

void export_Variant(pybind11::module& m)
    {
    pybind11::class_<Variant, VariantPy, std::shared_ptr<Variant> >(m,"Variant")
        .def(pybind11::init<>())
        .def("__call__", &Variant::operator())
        .def("min", &Variant::min)
        .def("max", &Variant::max)
        .def_property_readonly("range", &Variant::range)
        ;

    pybind11::class_<VariantConstant, Variant, std::shared_ptr<VariantConstant> >(m, "VariantConstant")
        .def(pybind11::init< Scalar >(), pybind11::arg("value"))
        .def_property("value", &VariantConstant::getValue, &VariantConstant::setValue)
        ;

    pybind11::class_<VariantRamp, Variant, std::shared_ptr<VariantRamp> >(m, "VariantRamp")
        .def(pybind11::init< Scalar, Scalar, uint64_t, uint64_t >(), pybind11::arg("A"),
                                                                     pybind11::arg("B"),
                                                                     pybind11::arg("t_start"),
                                                                     pybind11::arg("t_ramp"))
        .def_property("A", &VariantRamp::getA, &VariantRamp::setA)
        .def_property("B", &VariantRamp::getB, &VariantRamp::setB)
        .def_property("t_start", &VariantRamp::getTStart, &VariantRamp::setTStart)
        .def_property("t_ramp", &VariantRamp::getTRamp, &VariantRamp::setTRamp)
        ;

    pybind11::class_<VariantCycle, Variant, std::shared_ptr<VariantCycle> >(m, "VariantCycle")
        .def(pybind11::init< Scalar,
                             Scalar,
                             uint64_t,
                             uint64_t,
                             uint64_t,
                             uint64_t,
                             uint64_t >(),
            pybind11::arg("A"),
            pybind11::arg("B"),
            pybind11::arg("t_start"),
            pybind11::arg("t_A"),
            pybind11::arg("t_AB"),
            pybind11::arg("t_B"),
            pybind11::arg("t_BA"))


        .def_property("A", &VariantCycle::getA, &VariantCycle::setA)
        .def_property("B", &VariantCycle::getB, &VariantCycle::setB)
        .def_property("t_start", &VariantCycle::getTStart, &VariantCycle::setTStart)
        .def_property("t_A", &VariantCycle::getTA, &VariantCycle::setTA)
        .def_property("t_AB", &VariantCycle::getTAB, &VariantCycle::setTAB)
        .def_property("t_B", &VariantCycle::getTB, &VariantCycle::setTB)
        .def_property("t_BA", &VariantCycle::getTBA, &VariantCycle::setTBA)
        ;

    m.def("_test_variant_call", &testVariantCall);
    }
