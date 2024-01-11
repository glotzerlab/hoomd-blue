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
        .def_property("box", &VectorVariantBoxConstant::getBox, &VectorVariantBoxConstant::setBox)
        /*
        .def(
            pybind11::pickle(
                [](VectorVariantBoxConstant& variant)
                {
                    std::shared_ptr<BoxDim> box = variant.getBox();
                    std::array<Scalar, 6> arr = {box->getL().x,
                                                 box->getL().y,
                                                 box->getL().z,
                                                 box->getTiltFactorXY(),
                                                 box->getTiltFactorXZ(),
                                                 box->getTiltFactorYZ()};
                    return pybind11::make_tuple(arr);
                },
                [](pybind11::tuple params)
                {
                    Scalar Lx = params[0][0].cast<Scalar>();
                    Scalar Ly = params[0][1].cast<Scalar>();
                    Scalar Lz = params[0][2].cast<Scalar>();
                    Scalar xy = params[0][3].cast<Scalar>();
                    Scalar xz = params[0][4].cast<Scalar>();
                    Scalar yz = params[0][5].cast<Scalar>();
                    std::shared_ptr<BoxDim> box(Lx, Ly, Lz);
                    box->setTiltFactors(xy, xz, yz);
                    return VectorVariantBoxConstant(box);
                }));
                */
        ;

    pybind11::class_<VectorVariantBoxLinear,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxLinear>>(m, "VectorVariantBoxLinear")
        .def(pybind11::
                 init<std::shared_ptr<BoxDim>, std::shared_ptr<BoxDim>, std::shared_ptr<Variant>>())
        .def_property("initial_box",
                      &VectorVariantBoxLinear::getBox1,
                      &VectorVariantBoxLinear::setBox1)
        .def_property("final_box",
                      &VectorVariantBoxLinear::getBox2,
                      &VectorVariantBoxLinear::setBox2)
        .def_property("variant",
                      &VectorVariantBoxLinear::getVariant,
                      &VectorVariantBoxLinear::setVariant);

    pybind11::class_<VectorVariantBoxInverseVolumeRamp,
                     VectorVariantBox,
                     std::shared_ptr<VectorVariantBoxInverseVolumeRamp>>(
        m,
        "VectorVariantBoxInverseVolumeRamp")
        .def(pybind11::init<std::shared_ptr<BoxDim>, Scalar, uint64_t, uint64_t>())
        .def_property("box1",
                      &VectorVariantBoxInverseVolumeRamp::getBox1,
                      &VectorVariantBoxInverseVolumeRamp::setBox1)
        .def_property("t_start",
                      &VectorVariantBoxInverseVolumeRamp::getTStart,
                      &VectorVariantBoxInverseVolumeRamp::setTStart)
        .def_property("t_ramp",
                      &VectorVariantBoxInverseVolumeRamp::getTRamp,
                      &VectorVariantBoxInverseVolumeRamp::setTRamp);
    }

    } // end namespace detail

    } // end namespace hoomd
