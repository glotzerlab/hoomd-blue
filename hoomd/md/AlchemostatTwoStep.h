// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc


#ifndef __ALCHEMOSTAT_TWO_STEP__
#define __ALCHEMOSTAT_TWO_STEP__

#include "IntegrationMethodTwoStep.h"
#include "hoomd/AlchemyData.h"

// #include <pybind11/stl_bind.h>
// PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<AlchemicalMDParticle>>);
class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef)
        : IntegrationMethodTwoStep(sysdef, std::make_shared<ParticleGroup>()) {};
    virtual ~AlchemostatTwoStep() { }

    //! Get the number of degrees of freedom granted to a given group
    unsigned int getNDOF()
        {
        return getIntegraorNDOF() + m_alchemicalParticles.size();
        };

    virtual void randomizeVelocities(unsigned int timestep)
        {
        m_exec_conf->msg->warning()
            << "AlchMD hasn't implemented randomized velocities. Nothing done.";
        }

    //! Change the timestep
    void setDeltaT(Scalar deltaT)
        {
        m_deltaT = deltaT;
        m_halfDeltaT = Scalar(0.5) * m_deltaT * m_nTimeFactor;
        }

    //! Reinitialize the integration variables if needed (implemented in the actual subclasses)
    virtual void initializeIntegratorVariables() { }

    static unsigned int getIntegraorNDOF()
        {
        return 0;
        }

    virtual void setAlchemTimeFactor(unsigned int alchemTimeFactor)
        {
        m_nTimeFactor = alchemTimeFactor;
        m_halfDeltaT = Scalar(0.5) * m_deltaT * m_nTimeFactor;
        }

    std::vector<std::shared_ptr<AlchemicalMDParticle>> getAlchemicalParticleList()
        {
        return m_alchemicalParticles;
        }

    // virtual void setPreAlchemTime(unsigned int start_time_step)
    //     {
    //     m_nextAlchemTimeStep = start_time_step;
    //     m_force->setNextAlchemStep(start_time_step);
    //     m_pre = (start_time_step > 0) ? true : false;
    //     }

    protected:
    //!< A vector of all alchemical particles belonging to this integrator
    // TODO: make sure this is a synced list
    std::vector<std::shared_ptr<AlchemicalMDParticle>> m_alchemicalParticles;
    unsigned int m_nTimeFactor = 1; //!< Trotter factorization power
    Scalar m_halfDeltaT;            //!< The time step
    uint64_t m_nextAlchemTimeStep;
    std::string m_log_name;
    // TODO: general templating possible for two step methods?
    };

// TODO: base alchemostat methods need to be exported
// void export_AlchemostatTwoStep(pybind11::module& m)
//     {
//     // pybind11::bind_vector<std::vector<std::shared_ptr<AlchemicalMDParticle>>>(
//     //     m,
//     //     "AlchemicalParticlesList");

//     pybind11::class_<AlchemostatTwoStep,
//                      IntegrationMethodTwoStep,
//                      std::shared_ptr<AlchemostatTwoStep>>(m, "AlchemostatTwoStep")
//         .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
//         .def("setAlchemTimeFactor", &AlchemostatTwoStep::setAlchemTimeFactor)
//         .def("getAlchemicalParticleList", &AlchemostatTwoStep::getAlchemicalParticleList)
//         ;
//     }

#endif // #ifndef __ALCHEMOSTAT_TWO_STEP__
