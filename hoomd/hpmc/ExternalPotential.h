// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/SystemDefinition.h"

namespace hoomd
    {
namespace hpmc
    {
/** Functor that computes interactions of particles with external fields.

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

    /** Evaluate the energy of the external field interacting with one particle (translated box)

        @param type_i Type index of the particle.
        @param r_i Posiion of the particle in the box (un-shifted local particle).
        @param q_i Orientation of the particle
        @param charge_i Charge of the particle.
        @param trial Set to false when evaluating the energy of a current configuration. Set to
               true when evaluating a trial move.
        @returns Energy of the external interaction (possibly INFINITY).

        particleEnergy takes in positions that are in the current box coordinates. It will correct
        for the origin shift so that implemented potentials can work in the coordinate frame of the
        global unshifted box.
    */
    LongReal particleEnergy(unsigned int type_i,
                            const vec3<LongReal>& r_i,
                            const quat<LongReal>& q_i,
                            LongReal charge_i,
                            bool trial = true)
        {
        const auto& particle_data = m_sysdef->getParticleData();
        auto box = particle_data->getGlobalBox();

        auto shifted_r = r_i + vec3<LongReal>(particle_data->getOrigin());
        int3 tmp = make_int3(0, 0, 0);
        box.wrap(shifted_r, tmp);
        return particleEnergyImplementation(type_i, shifted_r, q_i, charge_i, trial);
        }

    /// Evaluate the total external energy due to this potential.
    LongReal totalEnergy(bool trial = false);

    protected:
    /// The system definition.
    std::shared_ptr<SystemDefinition> m_sysdef;

    /** Implement the evaluation the energy of the external field interacting with one particle.

        @param type_i Type index of the particle.
        @param r_i Posiion of the particle in the box.
        @param q_i Orientation of the particle
        @param charge_i Charge of the particle.
        @param trial Set to false when evaluating the energy of a current configuration. Set to
               true when evaluating a trial move.
        @returns Energy of the external interaction (possibly INFINITY).

        Note: Potentials that may return INFINITY should assume valid old configurations and return
        0 when trial is false. This avoids computing INFINITY - INFINITY -> NaN.

        HPMC tranlates all particles in the system, tracked by ParticleData::getOrigin.
        particleEnergy() removes this translation before calling particleEnergyImplmentation so that
        the implementation can compute the external energy as if the particle were in an
        un-translated box.
    */
    virtual LongReal particleEnergyImplementation(unsigned int type_i,
                                                  const vec3<LongReal>& r_i,
                                                  const quat<LongReal>& q_i,
                                                  LongReal charge_i,
                                                  bool trial)
        {
        return 0;
        }
    };

inline LongReal ExternalPotential::totalEnergy(bool trial)
    {
    LongReal total_energy = 0.0;

    const auto& particle_data = m_sysdef->getParticleData();

    ArrayHandle<Scalar4> h_postype(particle_data->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(particle_data->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar> h_charge(particle_data->getCharges(),
                                 access_location::host,
                                 access_mode::read);

    for (unsigned int i = 0; i < particle_data->getN(); i++)
        {
        Scalar4 postype_i = h_postype.data[i];
        auto r_i = vec3<LongReal>(postype_i);
        int type_i = __scalar_as_int(postype_i.w);
        auto q_i = quat<LongReal>(h_orientation.data[i]);

        total_energy += particleEnergy(type_i, r_i, q_i, h_charge.data[i], trial);
        }

#ifdef ENABLE_MPI
    if (particle_data->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &total_energy,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      particle_data->getExecConf()->getMPICommunicator());
        }
#endif

    return total_energy;
    }

    } // end namespace hpmc

    } // end namespace hoomd
