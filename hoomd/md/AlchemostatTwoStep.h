// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "AlchemyData.h"
#include "IntegrationMethodTwoStep.h"
#include "hoomd/filter/ParticleFilterNull.h"

#include <cstddef>
#include <pybind11/stl_bind.h>

namespace hoomd
    {

namespace md
    {

class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef, unsigned int alchemTimeFactor)
        : IntegrationMethodTwoStep(
              sysdef,
              std::make_shared<ParticleGroup>(sysdef, std::make_shared<ParticleFilterNull>())),
          m_nTimeFactor(alchemTimeFactor), m_halfDeltaT(0.5 * m_deltaT * alchemTimeFactor) { };

    virtual ~AlchemostatTwoStep() { }

    //! Get the number of degrees of freedom associated with the alchemostat
    unsigned int getNDOF()
        {
        return m_iteratorDOF + static_cast<unsigned int>(m_alchemicalParticles.size());
        };

    virtual void randomizeVelocities(unsigned int timestep)
        {
        m_exec_conf->msg->warning()
            << "AlchMD hasn't implemented randomized velocities. Nothing done.";
        }

    //! Change the timestep
    virtual void setDeltaT(Scalar deltaT)
        {
        IntegrationMethodTwoStep::setDeltaT(deltaT);
        m_halfDeltaT = Scalar(0.5) * m_deltaT * m_nTimeFactor;
        }

    //! Reinitialize the integration variables if needed (implemented in the actual subclasses)
    virtual void initializeIntegratorVariables() { }

    virtual void setAlchemTimeFactor(unsigned int alchemTimeFactor)
        {
        m_nTimeFactor = alchemTimeFactor;
        m_halfDeltaT = Scalar(0.5) * m_deltaT * m_nTimeFactor;
        }

    virtual unsigned int getAlchemTimeFactor()
        {
        return m_nTimeFactor;
        }

    std::vector<std::shared_ptr<AlchemicalMDParticle>>& getAlchemicalParticleList()
        {
        return m_alchemicalParticles;
        }

    void setAlchemicalParticleList(
        std::vector<std::shared_ptr<AlchemicalMDParticle>> alchemicalParticles)
        {
        m_alchemicalParticles.swap(alchemicalParticles);
        updateAlchemicalTimestep();
        }

    void setNextAlchemicalTimestep(unsigned int timestep)
        {
        m_nextAlchemTimeStep = timestep;
        updateAlchemicalTimestep();
        }

    void updateAlchemicalTimestep()
        {
        for (auto& alpha : m_alchemicalParticles)
            {
            alpha->m_nextTimestep = m_nextAlchemTimeStep;
            }
        }

    protected:
    //!< A vector of all alchemical particles belonging to this integrator
    std::vector<std::shared_ptr<AlchemicalMDParticle>> m_alchemicalParticles;
    unsigned int m_nTimeFactor = 1; //!< Trotter factorization power
    Scalar m_halfDeltaT;            //!< The time step
    uint64_t m_nextAlchemTimeStep;
    unsigned int m_iteratorDOF = 0;
    };

    } // end namespace md

    } // end namespace hoomd
