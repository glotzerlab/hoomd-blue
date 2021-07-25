// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMOSTAT_TWO_STEP__
#define __ALCHEMOSTAT_TWO_STEP__

#include "IntegrationMethodTwoStep.h"
#include "hoomd/AlchemyData.h"

#include <cstddef>
#include <pybind11/stl_bind.h>

//! all templating can be removed with c++20, virtual constexpr
// template<class T = void>
class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef)
        : IntegrationMethodTwoStep(sysdef, std::make_shared<ParticleGroup>()) {};
    virtual ~AlchemostatTwoStep() { }

    //! Get the number of degrees of freedom associated with the alchemostat
    unsigned int getNDOF()
        {
        return m_iteratorDOF + m_alchemicalParticles.size();
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
        
    virtual void getAlchemTimeFactor()
        {
        return m_nTimeFactor;
        }

    std::vector<std::shared_ptr<AlchemicalMDParticle>> getAlchemicalParticleList()
        {
        return m_alchemicalParticles;
        }

    void addAlchemicalParticle(std::shared_ptr<AlchemicalMDParticle> alpha)
        {
        m_alchemicalParticles.push_back(alpha);
        updateAlchemicalTimestep();
        }

    void setNextAlchemicalTimestep(unsigned int timestep)
        {
        assert(m_validState);
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

    void updateAlchemicalTimestep(uint64_t timestep)
        {
        for (auto& alpha : m_alchemicalParticles)
            {
            alpha->m_nextTimestep = timestep;
            }
        }

    protected:
    //!< A vector of all alchemical particles belonging to this integrator
    // TODO: make sure this is a synced list
    std::vector<std::shared_ptr<AlchemicalMDParticle>> m_alchemicalParticles;
    unsigned int m_nTimeFactor = 1; //!< Trotter factorization power
    Scalar m_halfDeltaT;            //!< The time step
    uint64_t m_nextAlchemTimeStep;
    unsigned int m_iteratorDOF = 0;
    bool m_validState = true; //!< Valid states are full alchemical timesteps
    // TODO: general templating possible for two step methods?
    };

// template<> inline unsigned int AlchemostatTwoStep<void>::getNDOF(){return 0;};

// TODO: base alchemostat methods need to be exported
inline void export_AlchemostatTwoStep(pybind11::module& m)
    {
    // pybind11::bind_vector<std::vector<std::shared_ptr<AlchemicalMDParticle>>>(
    //     m,
    //     "AlchemicalParticlesList");

    pybind11::class_<AlchemostatTwoStep,
                     IntegrationMethodTwoStep,
                     std::shared_ptr<AlchemostatTwoStep>>(m, "AlchemostatTwoStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setAlchemTimeFactor", &AlchemostatTwoStep::setAlchemTimeFactor)
        .def("getAlchemTimeFactor", &AlchemostatTwoStep::getAlchemTimeFactor)
        .def("getAlchemicalParticleList", &AlchemostatTwoStep::getAlchemicalParticleList)
        .def("addAlchemicalParticle", &AlchemostatTwoStep::addAlchemicalParticle)
        .def("getNDOF", &AlchemostatTwoStep::getNDOF)
        .def("setNextAlchemicalTimestep", &AlchemostatTwoStep::setNextAlchemicalTimestep);
    }

#endif // #ifndef __ALCHEMOSTAT_TWO_STEP__
