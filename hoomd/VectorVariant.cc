// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "VectorVariant.h"
#include <pybind11/stl.h>

namespace hoomd
    {
//* Trampoline for classes inherited in python
class VectorVariantBoxPy : public VectorVariantBox
    {
    public:
    // Inherit the constructors
    using VectorVariantBox::VectorVariantBox;

    // trampoline method
    array_type operator()(uint64_t timestep) override
        {
        PYBIND11_OVERLOAD_NAME(array_type,       // Return type
                               VectorVariantBox, // Parent class
                               "__call__",       // name of function in python
                               operator(),       // Name of function in C++
                               timestep          // Argument(s)
        );
        }
    };

namespace detail
    {

void export_VectorVariantBox(pybind11::module& m)
    {
    pybind11::class_<VectorVariantBox, VectorVariantBoxPy, std::shared_ptr<VectorVariantBox>>(
        m,
        "VectorVariantBox")
        .def(pybind11::init<>())
        .def("__call__", &VectorVariantBox::operator());

    pybind11::class_<VectorVariantBoxConstant,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxConstant>>(m, "VectorVariantBoxConstant")
        .def(pybind11::init<std::shared_ptr<BoxDim>>())
        .def_property("box", &VectorVariantBoxConstant::getBox, &VectorVariantBoxConstant::setBox);

    pybind11::class_<VectorVariantBoxLinear,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxLinear>>(m, "VectorVariantBoxLinear")
        .def(pybind11::init<std::shared_ptr<BoxDim>, std::shared_ptr<BoxDim>, uint64_t, uint64_t>())
        .def_property("box1", &VectorVariantBoxLinear::getBox1, &VectorVariantBoxLinear::setBox1)
        .def_property("box2", &VectorVariantBoxLinear::getBox2, &VectorVariantBoxLinear::setBox2)
        .def_property("t_start",
                      &VectorVariantBoxLinear::getTStart,
                      &VectorVariantBoxLinear::setTStart)
        .def_property("t_ramp",
                      &VectorVariantBoxLinear::getTRamp,
                      &VectorVariantBoxLinear::setTRamp);
    }

    } // end namespace detail

    } // end namespace hoomd
