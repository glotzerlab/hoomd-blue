// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AlchemyData.h"

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::md::AlchemicalMDParticle>>);

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AlchemicalMDParticles(pybind11::module& m)
    {
    pybind11::class_<AlchemicalMDParticle, std::shared_ptr<AlchemicalMDParticle>>(
        m,
        "AlchemicalMDParticle")
        .def_property("mass", &AlchemicalMDParticle::getMass, &AlchemicalMDParticle::setMass)
        .def_readwrite("mu", &AlchemicalMDParticle::mu)
        .def_readwrite("alpha", &AlchemicalMDParticle::value)
        .def_readwrite("momentum", &AlchemicalMDParticle::momentum)
        .def_property_readonly("forces", &AlchemicalMDParticle::getDAlphas)
        .def_property_readonly("net_force",
                               pybind11::overload_cast<>(&AlchemicalMDParticle::getNetForce))
        .def("notifyDetach", &AlchemicalMDParticle::notifyDetach);

    pybind11::class_<AlchemicalPairParticle,
                     AlchemicalMDParticle,
                     std::shared_ptr<AlchemicalPairParticle>>(m, "AlchemicalPairParticle");

    pybind11::class_<AlchemicalNormalizedPairParticle,
                     AlchemicalPairParticle,
                     std::shared_ptr<AlchemicalNormalizedPairParticle>>(
        m,
        "AlchemicalNormalizedPairParticle")
        .def_readwrite(
            "norm_value",
            &AlchemicalNormalizedPairParticle::alchemical_derivative_normalization_value);

    pybind11::bind_vector<std::vector<std::shared_ptr<hoomd::md::AlchemicalMDParticle>>>(
        m,
        "AlchemicalMDParticleList");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
