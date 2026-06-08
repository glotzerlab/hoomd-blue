// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialZetterling.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialZetterling::PairPotentialZetterling(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialZetterling::energy(const LongReal r_squared,
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
    LongReal invr = 1.0 / r;
    LongReal eval_cos = fast::cos(2.0 * param.kf * r);
    LongReal eval_exp = fast::exp(param.alpha * r);
    LongReal eval_invr_3 = invr * invr * invr;
    LongReal eval_sigma_over_r = param.sigma * invr;

    // Compute energy
    LongReal term1 = param.A * eval_exp * eval_invr_3 * eval_cos;
    LongReal term2 = param.B * fast::pow(eval_sigma_over_r, param.n);
    LongReal energy = term1 + term2;

    if (m_mode == shift || (m_mode == xplor && param.r_on_squared >= param.r_cut_squared))
        {
        LongReal r_cut = fast::sqrt(param.r_cut_squared);
        LongReal inv_r_cut = 1.0 / r_cut;
        LongReal r_cut_eval_cos = fast::cos(2.0 * param.kf * r_cut);
        LongReal r_cut_eval_exp = fast::exp(param.alpha * r_cut);
        LongReal r_cut_eval_invr_3 = inv_r_cut * inv_r_cut * inv_r_cut;
        LongReal r_cut_eval_sigma_over_r = param.sigma * inv_r_cut;

        // Compute energy
        LongReal r_cut_term1 = param.A * r_cut_eval_exp * r_cut_eval_invr_3 * r_cut_eval_cos;
        LongReal r_cut_term2 = param.B * fast::pow(r_cut_eval_sigma_over_r, param.n);
        energy -= r_cut_term1 + r_cut_term2;
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

void PairPotentialZetterling::setParamsPython(pybind11::tuple typ, pybind11::dict params)
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

pybind11::dict PairPotentialZetterling::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialZetterling(pybind11::module& m)
    {
    pybind11::class_<PairPotentialZetterling,
                     PairPotential,
                     std::shared_ptr<PairPotentialZetterling>>(m, "PairPotentialZetterling")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialZetterling::setParamsPython)
        .def("getParams", &PairPotentialZetterling::getParamsPython)
        .def_property("mode", &PairPotentialZetterling::getMode, &PairPotentialZetterling::setMode);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
