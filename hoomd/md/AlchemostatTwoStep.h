// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMOSTAT_TWO_STEP__
#define __ALCHEMOSTAT_TWO_STEP__

#include "AlchemyData.h"
#include "IntegrationMethodTwoStep.h"

#include <cstddef>
#include <pybind11/stl_bind.h>

namespace hoomd {

namespace md {

//! all templating can be removed with c++20, virtual constexpr
// template<class T = void>
class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef, unsigned int alchemTimeFactor)
        : IntegrationMethodTwoStep(sysdef, std::make_shared<ParticleGroup>()),
          m_nTimeFactor(alchemTimeFactor), m_halfDeltaT(0.5 * m_deltaT * alchemTimeFactor) {};

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

    std::vector<std::shared_ptr<AlchemicalMDParticle>> getAlchemicalParticleList()
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
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int>())
        .def_property("time_factor",
                      &AlchemostatTwoStep::getAlchemTimeFactor,
                      &AlchemostatTwoStep::setAlchemTimeFactor)
        .def_property("alchemical_particles",
                      &AlchemostatTwoStep::getAlchemicalParticleList,
                      &AlchemostatTwoStep::setAlchemicalParticleList)
        .def("getAlchemicalParticleList", &AlchemostatTwoStep::getAlchemicalParticleList)
        .def("getNDOF", &AlchemostatTwoStep::getNDOF)
        .def("setNextAlchemicalTimestep", &AlchemostatTwoStep::setNextAlchemicalTimestep);
    }

} // end namespace md

} // end namespace hoomd

#endif // #ifndef __ALCHEMOSTAT_TWO_STEP__
