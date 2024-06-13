// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialExpandedGaussian.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialExpandedGaussian::PairPotentialExpandedGaussian(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialExpandedGaussian::energy(const LongReal r_squared,
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

    LongReal r = fast::sqrt(r_squared);
    LongReal rmd_2 = (r - param.delta) * (r - param.delta);
    LongReal rmd_over_sigma_2 = rmd_2 / param.sigma_2;
    LongReal exp_val = fast::exp(-LongReal(1.0) / LongReal(2.0) * rmd_over_sigma_2);
    LongReal energy = param.epsilon * exp_val;

    if (m_mode == shift || (m_mode == xplor && param.r_on_squared >= param.r_cut_squared))
        {
        LongReal r_cut = fast::sqrt(param.r_cut_squared);
        LongReal rcutmd_2 = (r_cut - param.delta) * (r_cut - param.delta);
        LongReal rcutmd_over_sigma_2 = rcutmd_2 / param.sigma_2;
        energy -= param.epsilon * fast::exp(-LongReal(1.0) / LongReal(2.0) * rcutmd_over_sigma_2);
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

void PairPotentialExpandedGaussian::setParamsPython(pybind11::tuple typ, pybind11::dict params)
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

pybind11::dict PairPotentialExpandedGaussian::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialExpandedGaussian(pybind11::module& m)
    {
    pybind11::class_<PairPotentialExpandedGaussian,
                     PairPotential,
                     std::shared_ptr<PairPotentialExpandedGaussian>>(m, "PairPotentialExpandedGaussian")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialExpandedGaussian::setParamsPython)
        .def("getParams", &PairPotentialExpandedGaussian::getParamsPython)
        .def_property("mode",
                      &PairPotentialExpandedGaussian::getMode,
                      &PairPotentialExpandedGaussian::setMode);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
