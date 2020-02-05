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
    };

void export_Variant(pybind11::module& m)
    {
    pybind11::class_<Variant, VariantPy, std::shared_ptr<Variant> >(m,"Variant")
        .def(pybind11::init<>())
        .def("__call__", &Variant::operator())
        ;

    pybind11::class_<VariantConstant, Variant, std::shared_ptr<VariantConstant> >(m, "VariantConstant")
        .def(pybind11::init< Scalar >(), pybind11::arg("value"))
        .def_property("value", &VariantConstant::getValue, &VariantConstant::setValue)
        ;

    m.def("_test_variant_call", &testVariantCall);
    }
