// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jproc

#include "IntegrationMethodTwoStep.h"

#ifndef __ALCHEMOSTAT_TWO_STEP__
#define __ALCHEMOSTAT_TWO_STEP__

class AlchemostatTwoStep : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~AlchemostatTwoStep() {}

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF();

        virtual unsigned int getRotationalNDOF();

        virtual void randomizeVelocities(unsigned int timestep);

        //! Reinitialize the integration variables if needed (implemented in the actual subclasses)
        virtual void initializeIntegratorVariables() {}

    protected:
        std::shared_ptr<AlchParticles> m_alchParticles;    //!< A vector of all alchemical particles
        unsigned int m_nTStep;          //!< Trotter factorization power

// TODO: general templating possible for two step methods?

#endif // #ifndef __ALCHEMOSTAT_TWO_STEP__
