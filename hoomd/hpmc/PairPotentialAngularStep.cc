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
    //m_directors(sysdef->getParticleData()->getNTypes())
    {
    //unsigned int ntypes = m_sysdef->getParticleData()->getNTypes()
    //get patch index
    unsigned int patch_index = 
    m_directors.resize(patch_index)
    m_delta.resize(patch_index)
        if (!m_isotropic)
        {
            raise std::runtime_error("Could not pass in the isotropic potential.");
        }

    }

void PairPotentialAngularStep::setPatch(std::string patch_index, pybind11::object v)
    {
    unsigned int patch_index = 

    if (v.is_none())
        {
        m_directors[patch_index].clear();
        m_delta[patch_index].clear();
        return;
        }
    pybind11::list directors = v["directors"];
    pybind11::list deltas = v["deltas"];

    auto N = pybind11::len(m_directors);

    if (!deltas.is_none() && pybind11::len(deltas) != N)
        {
        throw std::runtime_error("the length of the delta list should match the length 
                                    of the director list.");
        }
    }

    m_directors[patch_index].resize(N);
    m_deltas[patch_index].resize(N);
    


// protected 
bool maskingFunction(const vec3< LongReal>& r_ij,
                    const unsigned int type_i, 
                    const quat<LongReal>& q_i,
                    const unsigned int type_j,
                    const quat<LongReal>& q_j)
    {

    //const auto& patch_m = m_patches[type_i]; 
    //const auto& patch_n = m_patches[type_j]; 

    LongReal cos_delta = cos(patch.delta);

    const vec3<LongReal> ehat_particle_reference_frame(1,0,0);
    vec3<LongReal> ehat_i = rotate(q_i, ehat_particle_reference_frame);
    vec3<LongReal> ehat_j = rotate(q_j, ehat_particle_reference_frame);

    LongReal rhat_ij = sqrtf(dot(r_ij, r_ij));

    for (int m = 0; m < m_directors[type_i].size(); m++) {
        for (int n = 0; n < m_directors[type_j].size(); n++) {
            if (dot(ehat_i, r_ij) >= cos_delta * rhat_ij
                && dot(ehat_j, -r_ij) >= cos_delta * rhat_ij)
                {
                return true;
                }
            else
                {
                return false;
                }
            }
        }
    }
    
virtual LongReal PairPotentialAngularStep::energy(const LongReal r_squared,
                                                 const vec3<LongReal>& r_ij,
                                                 const unsigned int type_i,
                                                 const quat<LongReal>& q_i,
                                                 const LongReal charge_i,
                                                 const unsigned int type_j,
                                                 const quat<LongReal>& q_j,
                                                 const LongReal charge_j) const
    {                 

    if (maskingFunction(r_ij, type_i, q_i, type_j, q_j)) //type_m and type_n
    {
        LongReal lj_energy = m_isotropic->energy(r_squared, r_ij, type_i, q_i, 
                                                 charge_i, type_j, q_j, charge_j);
        return lj_energy;
    }
    return 0; 

    }

/* -angular step potential: import the isotropic potential, if patch overlaps, the angular 
step potential is exactly how the isotropic potential behaves. if patch does not overlap at all, 
the angular potential is 0. 
-my understanding: we do not need to input the isotropic potential parameter, because we will get
it from elsewhere.
-need to modify this section. Figure out what getTypeByName does. 
- look into pybind11 documentation. 
*/

void PairPotentialAngularStep::setPatch()


void PairPotentialAngularStep::setParamsPython(pybind11::tuple typ, pybind11::dict params)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_m = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_n = pdata->getTypeByName(typ[1].cast<std::string>());
    m_patches[patch_m] = m_delta; // not sure how is delta represented here 
    m_patches[patch_n] = m_delta; 
    //auto type_i = pdata->getTypeByName(typ[0].cast<std::string>()); 
    //auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    //unsigned int param_index_1 = m_type_param_index(type_i, type_j);
    //m_param[param_index_1] = ParamType(params);
    //unsigned int param_index_2 = m_type_param_index(type_j, type_i);
    //m_param[param_index_2] = ParamType(params);
    }

pybind11::dict PairPotentialAngularStep::getParamsPython(pybind11::tuple typ)
    {
    auto pdata = m_sysdef->getParticleData();
    auto type_m = pdata->getTypeByName(typ[0].cast<std::string>());
    auto type_n = pdata->getTypeByName(typ[1].cast<std::string>());
    return m_patches[patch].asDict();
    //auto type_i = pdata->getTypeByName(typ[0].cast<std::string>());
    //auto type_j = pdata->getTypeByName(typ[1].cast<std::string>());
    //unsigned int param_index = m_type_param_index(type_i, type_j);
    //return m_params[param_index].asDict();
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
