// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevinBase.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/** Integrates part of the system forward in two steps with Langevin dynamics

    Langevin dynamics modifies standard NVE integration with two additional forces, a random force
   and a drag force. This implementation is very similar to TwoStepNVE with the additional forces.
   Note that this is not a really proper Langevin integrator, but it works well in practice.
*/
class PYBIND11_EXPORT TwoStepLangevin : public TwoStepLangevinBase
    {
    public:
    /** Constructs the integration method and associates it with the system
        @param sysdef SystemDefinition this method will act on. Must not be NULL.
        @param group The group of particles this integration method is to work on
        @param T Temperature set point as a function of time
    */
    TwoStepLangevin(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T);
    virtual ~TwoStepLangevin();

    /** Turn on or off Tally
        @param tally if true, tallies energy exchange from the thermal reservoir
    */
    void setTallyReservoirEnergy(bool tally)
        {
        m_tally = tally;
        }

    /// Get the tally setting
    bool getTallyReservoirEnergy()
        {
        return m_tally;
        }

    /// Get the reservoir energy
    Scalar getReservoirEnergy()
        {
        return m_reservoir_energy;
        }

    /// Performs the second step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    /// Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    /// The energy of the reservoir the system is coupled to.
    Scalar m_reservoir_energy;

    /// An energy packet that isn't added until the next time step
    Scalar m_extra_energy_overdeltaT;

    /// If true, changes to the energy of the reservoir are calculated
    bool m_tally;

    /// If set true, there will be no translational noise (random force)
    bool m_noiseless_t;

    /// If set true, there will be no rotational noise (random torque)
    bool m_noiseless_r;
    };

    } // end namespace md
    } // end namespace hoomd
