// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairYLZ.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

// Template specification for YLZ anisotropic pair potential. A specific
// template instance is needed since we expose the shape as just mu in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
inline void export_AnisoPotentialPair<EvaluatorPairYLZ>(pybind11::module& m,
                                                        const std::string& name)
    {
    pybind11::class_<AnisoPotentialPair<EvaluatorPairYLZ>,
                     ForceCompute,
                     std::shared_ptr<AnisoPotentialPair<EvaluatorPairYLZ>>>
        anisopotentialpair(m, name.c_str());
    anisopotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &AnisoPotentialPair<EvaluatorPairYLZ>::setParamsPython)
        .def("getParams", &AnisoPotentialPair<EvaluatorPairYLZ>::getParamsPython)
        .def("setMu", &AnisoPotentialPair<EvaluatorPairYLZ>::setShapePython)
        .def("getMu", &AnisoPotentialPair<EvaluatorPairYLZ>::getShapePython)
        .def("setRCut", &AnisoPotentialPair<EvaluatorPairYLZ>::setRCutPython)
        .def("getRCut", &AnisoPotentialPair<EvaluatorPairYLZ>::getRCut)
        .def_property("mode",
                      &AnisoPotentialPair<EvaluatorPairYLZ>::getShiftMode,
                      &AnisoPotentialPair<EvaluatorPairYLZ>::setShiftModePython)
        .def("getTypeShapesPy", &AnisoPotentialPair<EvaluatorPairYLZ>::getTypeShapesPy);
    }

void export_AnisoPotentialPairYLZ(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairYLZ>(m, "AnisoPotentialPairYLZ");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
