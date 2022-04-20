// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AnisoPotentialPair.h"
#include "EvaluatorPairDipole.h"

#ifdef ENABLE_HIP
#include "AnisoPotentialPairGPU.h"
#endif

namespace hoomd
    {
namespace md
    {

extern template class AnisoPotentialPair<EvaluatorPairDipole>;

#ifdef ENABLE_HIP
extern template class AnisoPotentialPairGPU<EvaluatorPairDipole>;
#endif

namespace detail
    {
// Template specification for Dipole anisotropic pair potential. A specific
// template instance is needed since we expose the shape as just mu in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
inline void export_AnisoPotentialPair<EvaluatorPairDipole>(pybind11::module& m,
                                                           const std::string& name)
    {
    pybind11::class_<AnisoPotentialPair<EvaluatorPairDipole>,
                     ForceCompute,
                     std::shared_ptr<AnisoPotentialPair<EvaluatorPairDipole>>>
        anisopotentialpair(m, name.c_str());
    anisopotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &AnisoPotentialPair<EvaluatorPairDipole>::setParamsPython)
        .def("getParams", &AnisoPotentialPair<EvaluatorPairDipole>::getParamsPython)
        .def("setMu", &AnisoPotentialPair<EvaluatorPairDipole>::setShapePython)
        .def("getMu", &AnisoPotentialPair<EvaluatorPairDipole>::getShapePython)
        .def("setRCut", &AnisoPotentialPair<EvaluatorPairDipole>::setRCutPython)
        .def("getRCut", &AnisoPotentialPair<EvaluatorPairDipole>::getRCut)
        .def_property("mode",
                      &AnisoPotentialPair<EvaluatorPairDipole>::getShiftMode,
                      &AnisoPotentialPair<EvaluatorPairDipole>::setShiftModePython)
        .def("slotWriteGSDShapeSpec",
             &AnisoPotentialPair<EvaluatorPairDipole>::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec", &AnisoPotentialPair<EvaluatorPairDipole>::connectGSDShapeSpec)
        .def("getTypeShapesPy", &AnisoPotentialPair<EvaluatorPairDipole>::getTypeShapesPy);
    }

extern template void export_AnisoPotentialPair<EvaluatorPairDipole>(pybind11::module& m,
                                                                    const std::string& name);

#ifdef ENABLE_HIP
extern template void export_AnisoPotentialPairGPU<EvaluatorPairDipole>(pybind11::module& m,
                                                                       const std::string& name);
#endif
    } // namespace detail

    } // end namespace md
    } // end namespace hoomd
