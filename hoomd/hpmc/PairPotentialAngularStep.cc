// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialAngularStep.h"

/* The idea is that user can define the number of patches on a particle via defining
the patch vectors. In addition, user can define the patch angles by providing a delta value 
that represents the half opening angles of the patch.
Is our conclusion from last meeting that there can only be multiple patch types on 1 particle?
*/

namespace hoomd
    {
namespace hpmc 
    { 

PairPotentialAngularStep::PairPotentialAngularStep(std::shared_ptr<SystemDefinition> sysdef, 
std::shared_ptr<PairPotential> isotropic)
    : PairPotential(sysdef), 
    m_isotropic(isotropic),
    m_patch(sysdef->getParticleData()->getNTypes())
    {
        if (!m_isotropic)
        {
            raise std::runtime_error("Could not pass in the isotropic potential.");
        }

    }

// protected 
bool maskingFunction(const vec3<LongReal>& r_ij,
                    const unsigned int type_i,
                    const quat<LongReal>& q_i,
                    const unsigned int type_j,
                    const quat<LongReal>& q_j)
    {

    const auto& patch_i = m_patch[type_i];
    const auto& patch_j = m_patch[type_j];

    LongReal cos_delta = cos(patch.delta);

    const vec3<LongReal> ehat_particle_reference_frame(1,0,0);
    vec3<LongReal> ehat_i = rotate(q_i, ehat_particle_reference_frame);
    vec3<LongReal> ehat_j = rotate(q_j, ehat_particle_reference_frame);

    LongReal r_ij_length = sqrtf(dot(r_ij, r_ij));

    if (dot(ehat_i, r_ij) >= cos_delta * r_ij_length
        && dot(ehat_j, -r_ij) >= cos_delta * r_ij_length)
        {
        return true;
        }
    else
        {
        return false;
        }
    }
    
LongReal PairPotentialAngularStep::energy(const LongReal r_squared,
                                                 const vec3<LongReal>& r_ij,
                                                 const unsigned int type_i,
                                                 const quat<LongReal>& q_i,
                                                 const LongReal charge_i,
                                                 const unsigned int type_j,
                                                 const quat<LongReal>& q_j,
                                                 const LongReal charge_j) const
    {                 

    if (maskingFunction(r_ij, type_i, q_i, type_j, q_j))
    {
        LongReal lj_energy = m_isotropic->energy(r_squared, r_ij, type_i, q_i, 
                                                 charge_i, type_j, q_j, charge_j);
        return lj_energy;
    }
    return 0; 

    }

void PairPotentialAngularStep::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index_1 = m_type_param_index(type_i, type_j);
    m_param[param_index_1] = ParamType(params);
    unsigned int param_index_2 = m_type_param_index(type_j, type_i);
    m_param[param_index_2] = ParamType(params);
    }

pybind11::dict PairPotentialAngularStep::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    unsigned int param_index = m_type_param_index(type_i, type_j);
    return m_params[param_index].asDict();
    }

namespace detail
    {
void exportPairPotentialAngularStep(pybind11::module& m)
    {
    pybind11::class_<PairPotentialAngularStep,
                     PairPotential,
                     std::shared_ptr<PairPotentialAngularStep>>(m, "PairPotentialAngularStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<PairPotential>>())
        .def("setParams", &PairPotentialAngularStep::setParamsPython)
        .def("getParams", &PairPotentialAngularStep::getParamsPython)
        .def_property("delta", &PairPotentialAngularStep::getDelta,
                      &PairPotentialAngularStep::setDelta) 
    }
    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd



// 
