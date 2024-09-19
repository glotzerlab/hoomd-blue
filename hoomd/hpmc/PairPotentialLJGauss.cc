// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialLJGauss.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialLJGauss::PairPotentialLJGauss(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialLJGauss::energy(const LongReal r_squared,
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
    LongReal rdiff = r - param.r0;
    LongReal rdiff_sigma_2 = rdiff / param.sigma_2;
    LongReal exp_val = fast::exp(-LongReal(0.5) * rdiff_sigma_2 * rdiff);
    LongReal r2_inverse = LongReal(1.0) / r_squared;
    LongReal r6_inverse = r2_inverse * r2_inverse * r2_inverse;

    LongReal energy = r6_inverse * (r6_inverse - LongReal(2.0)) - exp_val * param.epsilon;

    if (m_mode == shift || (m_mode == xplor && param.r_on_squared >= param.r_cut_squared))
        {
        LongReal r_cut_2_inverse = LongReal(1.0) / param.r_cut_squared;
        LongReal r_cut_6_inverse = r_cut_2_inverse * r_cut_2_inverse * r_cut_2_inverse;
        LongReal r_cut_minus_r0 = fast::sqrt(param.r_cut_squared) - param.r0;

        energy 
            -= r_cut_6_inverse * (r_cut_6_inverse - LongReal(2.0))
                 - (param.epsilon * fast::exp(-LongReal(0.5) * r_cut_minus_r0 * r_cut_minus_r0 / param.sigma_2));

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

void PairPotentialLJGauss::setParamsPython(pybind11::tuple typ, pybind11::dict params)
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

pybind11::dict PairPotentialLJGauss::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialLJGauss(pybind11::module& m)
    {
    pybind11::class_<PairPotentialLJGauss,
                     PairPotential,
                     std::shared_ptr<PairPotentialLJGauss>>(m, "PairPotentialLJGauss")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialLJGauss::setParamsPython)
        .def("getParams", &PairPotentialLJGauss::getParamsPython)
        .def_property("mode",
                      &PairPotentialLJGauss::getMode,
                      &PairPotentialLJGauss::setMode);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
