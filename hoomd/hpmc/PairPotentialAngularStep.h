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

    bool maskingFunction(const vec3<LongReal>& r_ij,
                        const unsigned int type_i,
                        const quat<LongReal>& q_i,
                        const unsigned int type_j,
                        const quat<LongReal>& q_j);


    virtual LongReal energy(const LongReal r_squared,
                            const vec3<LongReal>& r_ij,
                            const unsigned int type_i,
                            const quat<LongReal>& q_i,
                            const LongReal charge_i,
                            const unsigned int type_j,
                            const quat<LongReal>& q_j,
                            const LongReal charge_j) const;


    /// Set type pair dependent parameters to the potential.
    void setParamsPython(pybind11::tuple typ, pybind11::dict params);

    /// Get type pair dependent parameters.
    pybind11::dict getParamsPython(pybind11::tuple typ);
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

