// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairChain.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

// Template specification for Chain anisotropic pair potential. A specific
// template instance is needed since we expose the shape as just mu in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
inline void export_AnisoPotentialPair<EvaluatorPairChain>(pybind11::module& m,
                                                           const std::string& name)
    {
    pybind11::class_<AnisoPotentialPair<EvaluatorPairChain>,
                     ForceCompute,
                     std::shared_ptr<AnisoPotentialPair<EvaluatorPairChain>>>
        anisopotentialpair(m, name.c_str());
    anisopotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &AnisoPotentialPair<EvaluatorPairChain>::setParamsPython)
        .def("getParams", &AnisoPotentialPair<EvaluatorPairChain>::getParamsPython)
        .def("setMu", &AnisoPotentialPair<EvaluatorPairChain>::setShapePython)
        .def("getMu", &AnisoPotentialPair<EvaluatorPairChain>::getShapePython)
        .def("setRCut", &AnisoPotentialPair<EvaluatorPairChain>::setRCutPython)
        .def("getRCut", &AnisoPotentialPair<EvaluatorPairChain>::getRCut)
        .def_property("mode",
                      &AnisoPotentialPair<EvaluatorPairChain>::getShiftMode,
                      &AnisoPotentialPair<EvaluatorPairChain>::setShiftModePython)
        .def("getTypeShapesPy", &AnisoPotentialPair<EvaluatorPairChain>::getTypeShapesPy);
    }

void export_AnisoPotentialPairChain(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairChain>(m, "AnisoPotentialPairChain");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
