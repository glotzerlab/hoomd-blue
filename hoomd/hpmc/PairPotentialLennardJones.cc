// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialLennardJones.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialLennardJones::PairPotentialLennardJones(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialLennardJones::energy(const LongReal r_squared,
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

    LongReal lj2 = param.epsilon_x_4 * param.sigma_6;
    LongReal lj1 = lj2 * param.sigma_6;

    LongReal r_2_inverse = LongReal(1.0) / r_squared;
    LongReal r_6_inverse = r_2_inverse * r_2_inverse * r_2_inverse;

    LongReal energy = r_6_inverse * (lj1 * r_6_inverse - lj2);

    if (m_mode == shift || (m_mode == xplor && param.r_on_squared >= param.r_cut_squared))
        {
        LongReal r_cut_2_inverse = LongReal(1.0) / param.r_cut_squared;
        LongReal r_cut_6_inverse = r_cut_2_inverse * r_cut_2_inverse * r_cut_2_inverse;
        energy -= r_cut_6_inverse * (lj1 * r_cut_6_inverse - lj2);
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

void PairPotentialLennardJones::setParamsPython(pybind11::tuple typ, pybind11::dict params)
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

pybind11::dict PairPotentialLennardJones::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialLennardJones(pybind11::module& m)
    {
    pybind11::class_<PairPotentialLennardJones,
                     PairPotential,
                     std::shared_ptr<PairPotentialLennardJones>>(m, "PairPotentialLennardJones")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialLennardJones::setParamsPython)
        .def("getParams", &PairPotentialLennardJones::getParamsPython)
        .def_property("mode",
                      &PairPotentialLennardJones::getMode,
                      &PairPotentialLennardJones::setMode);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
