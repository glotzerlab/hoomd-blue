// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Variant.h"

namespace hoomd
    {
//* Trampoline for classes inherited in python
class VariantPy : public Variant
    {
    public:
    // Inherit the constructors
    using Variant::Variant;

    // trampoline method
    Scalar operator()(uint64_t timestep) override
        {
        PYBIND11_OVERLOAD_NAME(Scalar,     // Return type
                               Variant,    // Parent class
                               "__call__", // name of function in python
                               operator(), // Name of function in C++
                               timestep    // Argument(s)
        );
        }

    Scalar min() override
        {
        PYBIND11_OVERLOAD_PURE_NAME(Scalar,  // Return type
                                    Variant, // Parent class
                                    "_min",  // name of function in python
                                    min      // name of function
        );
        }

    Scalar max() override
        {
        PYBIND11_OVERLOAD_PURE_NAME(Scalar,  // Return type
                                    Variant, // Parent class
                                    "_max",  // name of function in python
                                    max      // name of function
        );
        }
    };

namespace detail
    {
// These testVariant{Method} functions allow us to test that Python custom
// variants work properly in C++. This ensures we can test that the function
// itself can be called in C++ when defined in Python.

/// Method to enable unit testing of C++ variant calls from pytest
Scalar testVariantCall(std::shared_ptr<Variant> t, uint64_t step)
    {
    return (*t)(step);
    }

/// Method to enable unit testing of C++ variant min class from pytest
Scalar testVariantMin(std::shared_ptr<Variant> t)
    {
    return t->min();
    }

/// Method to enable unit testing of C++ variant max class from pytest
Scalar testVariantMax(std::shared_ptr<Variant> t)
    {
    return t->max();
    }

void export_Variant(pybind11::module& m)
    {
    pybind11::class_<Variant, VariantPy, std::shared_ptr<Variant>>(m, "Variant")
        .def(pybind11::init<>())
        .def("__call__", &Variant::operator())
        .def("_min", &Variant::min)
        .def("_max", &Variant::max)
        .def_property_readonly("range", &Variant::range);

    pybind11::class_<VariantConstant, Variant, std::shared_ptr<VariantConstant>>(m,
                                                                                 "VariantConstant")
        .def(pybind11::init<Scalar>(), pybind11::arg("value"))
        .def_property("value", &VariantConstant::getValue, &VariantConstant::setValue)
        .def(pybind11::pickle(
            [](const VariantConstant& variant) { return pybind11::make_tuple(variant.getValue()); },
            [](pybind11::tuple params) { return VariantConstant(params[0].cast<Scalar>()); }));

    pybind11::class_<VariantRamp, Variant, std::shared_ptr<VariantRamp>>(m, "VariantRamp")
        .def(pybind11::init<Scalar, Scalar, uint64_t, uint64_t>(),
             pybind11::arg("A"),
             pybind11::arg("B"),
             pybind11::arg("t_start"),
             pybind11::arg("t_ramp"))
        .def_property("A", &VariantRamp::getA, &VariantRamp::setA)
        .def_property("B", &VariantRamp::getB, &VariantRamp::setB)
        .def_property("t_start", &VariantRamp::getTStart, &VariantRamp::setTStart)
        .def_property("t_ramp", &VariantRamp::getTRamp, &VariantRamp::setTRamp)
        .def(pybind11::pickle(
            [](const VariantRamp& variant)
            {
                return pybind11::make_tuple(variant.getA(),
                                            variant.getB(),
                                            variant.getTStart(),
                                            variant.getTRamp());
            },
            [](pybind11::tuple params)
            {
                return VariantRamp(params[0].cast<Scalar>(),
                                   params[1].cast<Scalar>(),
                                   params[2].cast<uint64_t>(),
                                   params[3].cast<uint64_t>());
            }));

    pybind11::class_<VariantCycle, Variant, std::shared_ptr<VariantCycle>>(m, "VariantCycle")
        .def(pybind11::init<Scalar, Scalar, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>(),
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
        .def(pybind11::pickle(
            [](const VariantCycle& variant)
            {
                return pybind11::make_tuple(variant.getA(),
                                            variant.getB(),
                                            variant.getTStart(),
                                            variant.getTA(),
                                            variant.getTAB(),
                                            variant.getTB(),
                                            variant.getTBA());
            },
            [](pybind11::tuple params)
            {
                return VariantCycle(params[0].cast<Scalar>(),
                                    params[1].cast<Scalar>(),
                                    params[2].cast<uint64_t>(),
                                    params[3].cast<uint64_t>(),
                                    params[4].cast<uint64_t>(),
                                    params[5].cast<uint64_t>(),
                                    params[6].cast<uint64_t>());
            }));

    pybind11::class_<VariantPower, Variant, std::shared_ptr<VariantPower>>(m, "VariantPower")
        .def(pybind11::init<Scalar, Scalar, double, uint64_t, uint64_t>(),
             pybind11::arg("A"),
             pybind11::arg("B"),
             pybind11::arg("power"),
             pybind11::arg("t_start"),
             pybind11::arg("t_ramp"))
        .def_property("A", &VariantPower::getA, &VariantPower::setA)
        .def_property("B", &VariantPower::getB, &VariantPower::setB)
        .def_property("power", &VariantPower::getPower, &VariantPower::setPower)
        .def_property("t_start", &VariantPower::getTStart, &VariantPower::setTStart)
        .def_property("t_ramp", &VariantPower::getTRamp, &VariantPower::setTRamp)
        .def(pybind11::pickle(
            [](const VariantPower& variant)
            {
                return pybind11::make_tuple(variant.getA(),
                                            variant.getB(),
                                            variant.getPower(),
                                            variant.getTStart(),
                                            variant.getTRamp());
            },
            [](pybind11::tuple params)
            {
                return VariantPower(params[0].cast<Scalar>(),
                                    params[1].cast<Scalar>(),
                                    params[2].cast<double>(),
                                    params[3].cast<uint64_t>(),
                                    params[4].cast<uint64_t>());
            }));

    m.def("_test_variant_call", &testVariantCall);
    m.def("_test_variant_min", &testVariantMin);
    m.def("_test_variant_max", &testVariantMax);
    }

    } // end namespace detail

    } // end namespace hoomd
