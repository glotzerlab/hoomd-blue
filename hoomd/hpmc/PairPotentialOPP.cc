// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialOPP.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialOPP::PairPotentialOPP(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialOPP::energy(const LongReal r_squared,
                                  const vec3<LongReal>& r_ij,
                                  const unsigned int type_i,
                                  const quat<LongReal>& q_i,
                                  const LongReal charge_i,
                                  const unsigned int type_j,
                                  const quat<LongReal>& q_j,
                                  const LongReal charge_j) const
    {
    unsigned int param_index = m_type_param_index(type_i, type_j);
    const auto& param = m_params[param_index];

    // Get quantities need for both energy calculation
    LongReal r = fast::sqrt(r_squared);
    LongReal eval_cos = fast::cos(param.k * r - param.phi);

    // Compute energy
    LongReal r_eta1_arg = param.C1 * fast::pow(r, -param.eta1);
    LongReal r_to_eta2 = fast::pow(r, -param.eta2);
    LongReal r_eta2_arg = param.C2 * r_to_eta2 * eval_cos;
    LongReal energy = r_eta1_arg + r_eta2_arg;

    if (m_mode == shift || (m_mode == xplor && param.r_on_squared >= param.r_cut_squared))
        {
        LongReal r_cut = fast::sqrt(param.r_cut_squared);
        LongReal r_cut_eta1_arg = param.C1 * fast::pow(r_cut, -param.eta1);
        LongReal r_cut_eta2_arg
            = param.C2 * fast::pow(r_cut, -param.eta2) * fast::cos(param.k * r_cut - param.phi);
        energy -= r_cut_eta1_arg + r_cut_eta2_arg;
        }

    if (m_mode == xplor && r_squared > param.r_on_squared)
        {
        LongReal a = param.r_cut_squared - param.r_on_squared;
        LongReal denominator = a * a * a;

        LongReal b = param.r_cut_squared - r_squared;
        LongReal numerator = b * b
                             * (param.r_cut_squared + LongReal(2.0) * r_squared
                                - LongReal(3.0) * param.r_on_squared);
        energy *= numerator / denominator;
        }

    return energy;
    }

void PairPotentialOPP::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index_1 = m_type_param_index(type_i, type_j);
    m_params[param_index_1] = ParamType(params);
    unsigned int param_index_2 = m_type_param_index(type_j, type_i);
    m_params[param_index_2] = ParamType(params);

    notifyRCutChanged();
    }

pybind11::dict PairPotentialOPP::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialOPP(pybind11::module& m)
    {
    pybind11::class_<PairPotentialOPP, PairPotential, std::shared_ptr<PairPotentialOPP>>(
        m,
        "PairPotentialOPP")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialOPP::setParamsPython)
        .def("getParams", &PairPotentialOPP::getParamsPython)
        .def_property("mode", &PairPotentialOPP::getMode, &PairPotentialOPP::setMode);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
