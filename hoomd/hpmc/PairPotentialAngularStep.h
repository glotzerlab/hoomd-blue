// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/hpmc/IntegratorHPMC.h"
#include "PairPotentialLennardJones.h"

namespace hoomd 
    {
namespace hpmc
    {

/*** Compute angular step potential between two particles.

For use with HPMC simulations. 
*/
class PairPotentialAngularStep : public hpmc::PairPotential
    {
    public:
    PairPotentialAngularStep(std::shared_ptr<SystemDefinition> sysdef, 
    std::shared_ptr<PairPotential> isotropic);
    virtual ~PairPotentialAngularStep() { }

    void setPatch(std::string patch_index, pybind11:object v);

    pybind11::object getPatch(std::string patch_index);

    bool maskingFunction(const vec3<LongReal>& r_ij,
                        const unsigned int type_i,
                        const quat<LongReal>& q_i,
                        const unsigned int type_j,
                        const quat<LongReal>& q_j);

    /*** Evaluate the energy of the pair interaction
        @param r_squared Pre-computed dot(r_ij, r_ij).
        @param r_ij Vector pointing from particle i to j.
        @param type_i Integer type index of particle i.
        @param charge_i Charge of particle i.
        @param q_i Orientation quaternion of particle i.
        @param type_j Integer type index of particle j.
        @param q_j Orientation quaternion of particle j.
        @param charge_j Charge of particle j.
        @returns Energy of the pair interaction.
    */

    virtual LongReal energy(const LongReal r_squared,
                            const vec3<LongReal>& r_ij,
                            const unsigned int type_i,
                            const quat<LongReal>& q_i,
                            const LongReal charge_i,
                            const unsigned int type_j,
                            const quat<LongReal>& q_j,
                            const LongReal charge_j) const;

    virtual LongReal computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
        {
        // Pass on the non-additive r_cut from the isotropic (parent) potential.
        return m_isotropic_potential->computeRCutNonAdditive(type_i, type_j);
        }
    
    void setDelta(LongReal delta){m_delta = delta;}
    LongReal getDelta(){return delta;}

    protected:
    /// Create a pointer to get the isotropic pair potential
    std::shared_ptr<PairPotentialLennardJones> m_isotropic;

    /// Type pair parameters of potential
    std:vector<std::vector<vec3<LongReal>>> m_directors;
    <std::vector<vec3<LongReal>> m_delta;

    };

    } // end namespace hpmc
    } // end namespace hoomd

