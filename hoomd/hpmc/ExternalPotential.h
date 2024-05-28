// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/SystemDefinition.h"

namespace hoomd
    {
namespace hpmc
    {
/*** Functor that computes interactions of particles with external fields.

    ExternalPotential allows energetic interactions to be included in an HPMC simulation. This
    abstract base class defines the API for the external energy object, consisting of a the energy
    evaluation fuction.

    Provide a ExternalPotential instance to IntegratorHPMC. The external potential energy will be
    evaluated when needed during the HPMC trial moves.
*/
class ExternalPotential
    {
    public:
    ExternalPotential(std::shared_ptr<SystemDefinition> sysdef) : m_sysdef(sysdef) { }
    virtual ~ExternalPotential() { }

    /*** Evaluate the energy of the external field interacting with one particle

        @param box Simulation box.
        @param type_i Type index of the particle.
        @param r_i Posiion of the particle in the box.
        @param q_i Orientation of the particle
        @param charge Charge of the particle.
        @param trial Set to false when evaluating the energy of a current configuration. Set to
               true when evaluating a trial move.
        @returns Energy of the external interaction (possibly INFINITY).

        Note: Potentials that may return INFINITY should assume valid old configurations and return
        0 when trial is false. This avoids computing INFINITY - INFINITY -> NaN.
    */
    virtual LongReal particleEnergy(const BoxDim& box,
                                    unsigned int type_i,
                                    const vec3<LongReal>& r_i,
                                    const quat<LongReal>& q_i,
                                    LongReal charge_i,
                                    bool trial = true)
        {
        return 0;
        }

    protected:
    /// The system definition.
    std::shared_ptr<SystemDefinition> m_sysdef;
    };

    } // end namespace hpmc

    } // end namespace hoomd
