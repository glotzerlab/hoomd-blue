// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairOriQuench.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

// Template specification for OriQuench anisotropic pair potential. A specific
// template instance is needed since we expose the shape as just mu in Python
// when the default behavior exposes setting and getting the shape through
// 'shape'.
template<>
inline void export_AnisoPotentialPair<EvaluatorPairOriQuench>(pybind11::module& m,
                                                        const std::string& name)
    {
    pybind11::class_<AnisoPotentialPair<EvaluatorPairOriQuench>,
                     ForceCompute,
                     std::shared_ptr<AnisoPotentialPair<EvaluatorPairOriQuench>>>
        anisopotentialpair(m, name.c_str());
    anisopotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setParams", &AnisoPotentialPair<EvaluatorPairOriQuench>::setParamsPython)
        .def("getParams", &AnisoPotentialPair<EvaluatorPairOriQuench>::getParamsPython)
        .def("setMu", &AnisoPotentialPair<EvaluatorPairOriQuench>::setShapePython)
        .def("getMu", &AnisoPotentialPair<EvaluatorPairOriQuench>::getShapePython)
        .def("setRCut", &AnisoPotentialPair<EvaluatorPairOriQuench>::setRCutPython)
        .def("getRCut", &AnisoPotentialPair<EvaluatorPairOriQuench>::getRCut)
        .def_property("mode",
                      &AnisoPotentialPair<EvaluatorPairOriQuench>::getShiftMode,
                      &AnisoPotentialPair<EvaluatorPairOriQuench>::setShiftModePython)
        .def("getTypeShapesPy", &AnisoPotentialPair<EvaluatorPairOriQuench>::getTypeShapesPy);
    }

void export_AnisoPotentialPairOriQuench(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOriQuench>(m, "AnisoPotentialPairOriQuench");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
