// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/md/PotentialExternal.h"
#include "hoomd/md/EvaluatorExternal@_evaluator@.h"
#include "hoomd/md/EvaluatorExternalElectricField.h"

namespace hoomd { namespace md { namespace detail {

// Export setParamsPython and getParams as a different name
// Electric field only has one parameter, so we can get its parameter from
// python with by a name other than getParams and setParams
template<> inline
void export_PotentialExternal<EvaluatorExternalElectricField>(
    pybind11::module& m,
    const std::string& name)
    {
    pybind11::class_<PotentialExternal<EvaluatorExternalElectricField>,
                     ForceCompute,
                     std::shared_ptr<PotentialExternal<EvaluatorExternalElectricField>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setE", &PotentialExternal<EvaluatorExternalElectricField>::setParamsPython)
        .def("getE", &PotentialExternal<EvaluatorExternalElectricField>::getParams);
    }

void export_PotentialExternal@_evaluator@(pybind11::module &m)
    {
    export_PotentialExternal<EvaluatorExternal@_evaluator@>(m, "PotentialExternal@_evaluator@");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
