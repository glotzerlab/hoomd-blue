// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

// This testVariantCall function allows us to test that Python custom vector
// variants work properly in C++. This ensures we can test that the function
// itself can be called in C++ when defined in Python.

/// Method to enable unit testing of C++ variant calls from pytest
std::array<Scalar, 6> testVectorVariantBoxCall(std::shared_ptr<VectorVariantBox> t, uint64_t step)
    {
    return (*t)(step);
    }

void export_VectorVariantBoxClasses(pybind11::module& m)
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
        .def_property("_box", &VectorVariantBoxConstant::getBox, &VectorVariantBoxConstant::setBox);

    pybind11::class_<VectorVariantBoxInterpolate,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxInterpolate>>(m, "VectorVariantBoxInterpolate")
        .def(pybind11::
                 init<std::shared_ptr<BoxDim>, std::shared_ptr<BoxDim>, std::shared_ptr<Variant>>())
        .def_property("_initial_box",
                      &VectorVariantBoxInterpolate::getInitialBox,
                      &VectorVariantBoxInterpolate::setInitialBox)
        .def_property("_final_box",
                      &VectorVariantBoxInterpolate::getFinalBox,
                      &VectorVariantBoxInterpolate::setFinalBox)
        .def_property("variant",
                      &VectorVariantBoxInterpolate::getVariant,
                      &VectorVariantBoxInterpolate::setVariant);

    pybind11::class_<VectorVariantBoxInverseVolumeRamp,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxInverseVolumeRamp>>(
        m,
        "VectorVariantBoxInverseVolumeRamp")
        .def(pybind11::init<std::shared_ptr<BoxDim>, Scalar, uint64_t, uint64_t>())
        .def_property("_initial_box",
                      &VectorVariantBoxInverseVolumeRamp::getInitialBox,
                      &VectorVariantBoxInverseVolumeRamp::setInitialBox)
        .def_property("t_start",
                      &VectorVariantBoxInverseVolumeRamp::getTStart,
                      &VectorVariantBoxInverseVolumeRamp::setTStart)
        .def_property("t_ramp",
                      &VectorVariantBoxInverseVolumeRamp::getTRamp,
                      &VectorVariantBoxInverseVolumeRamp::setTRamp)
        .def_property("final_volume",
                      &VectorVariantBoxInverseVolumeRamp::getFinalVolume,
                      &VectorVariantBoxInverseVolumeRamp::setFinalVolume);

    m.def("_test_vector_variant_box_call", &testVectorVariantBoxCall);
    }

    } // end namespace detail

    } // end namespace hoomd
