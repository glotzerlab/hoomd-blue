// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialStep.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialStep::PairPotentialStep(std::shared_ptr<SystemDefinition> sysdef)
    : PairPotential(sysdef), m_params(m_type_param_index.getNumElements())
    {
    }

LongReal PairPotentialStep::computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
    {
    unsigned int param_index = m_type_param_index(type_i, type_j);
    size_t n = m_params[param_index].m_r_squared.size();
    if (n > 0)
        {
        return slow::sqrt(m_params[param_index].m_r_squared[n - 1]);
        }
    else
        {
        return 0;
        }
    }

LongReal PairPotentialStep::energy(const LongReal r_squared,
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

    size_t N = param.m_epsilon.size();

    if (N == 0)
        {
        return 0;
        }

    if (r_squared < param.m_r_squared[0])
        {
        return param.m_epsilon[0];
        }

    // Perform a binary search based on r_squared to find the relevant potential value.
    ssize_t L = 0;
    ssize_t R = N;

    while (L < R)
        {
        size_t m = (L + R) / 2;
        LongReal r_squared_m = param.m_r_squared[m];

        if (r_squared_m <= r_squared)
            {
            L = m + 1;
            }
        else
            {
            R = m;
            }
        }

    if (size_t(L) < N)
        {
        return param.m_epsilon[L];
        }
    else
        {
        return 0;
        }
    }

void PairPotentialStep::setParamsPython(pybind11::tuple typ, pybind11::object params)
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

pybind11::dict PairPotentialStep::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

PairPotentialStep::ParamType::ParamType(pybind11::object params)
    {
    if (params.is_none())
        {
        m_epsilon.clear();
        m_r_squared.clear();
        return;
        }

    pybind11::dict v = params;
    pybind11::list epsilon_list = v["epsilon"];
    pybind11::list r_list = v["r"];

    auto N = pybind11::len(epsilon_list);
    if (pybind11::len(r_list) != N)
        {
        throw std::runtime_error("Both epsilon and r must have the same length.");
        }

    m_epsilon.resize(N);
    m_r_squared.resize(N);

    for (unsigned int i = 0; i < N; i++)
        {
        m_epsilon[i] = epsilon_list[i].cast<LongReal>();
        LongReal r = r_list[i].cast<LongReal>();
        m_r_squared[i] = r * r;

        if (i >= 1 && m_r_squared[i] <= m_r_squared[i - 1])
            {
            throw std::domain_error("r must monotonically increase.");
            }
        }
    }

pybind11::dict PairPotentialStep::ParamType::asDict()
    {
    size_t N = m_epsilon.size();
    pybind11::list epsilon;
    pybind11::list r;

    for (unsigned int i = 0; i < N; i++)
        {
        epsilon.append(m_epsilon[i]);
        r.append(slow::sqrt(m_r_squared[i]));
        }

    pybind11::dict result;
    result["epsilon"] = epsilon;
    result["r"] = r;

    return result;
    }

namespace detail
    {
void exportPairPotentialStep(pybind11::module& m)
    {
    pybind11::class_<PairPotentialStep, PairPotential, std::shared_ptr<PairPotentialStep>>(
        m,
        "PairPotentialStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PairPotentialStep::setParamsPython)
        .def("getParams", &PairPotentialStep::getParamsPython);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
