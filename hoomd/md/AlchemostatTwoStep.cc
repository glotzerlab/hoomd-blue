// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "AlchemostatTwoStep.h"

using namespace std;
// namespace py = pybind11;

AlchemostatTwoStep::AlchemostatTwoStep(std::shared_ptr<SystemDefinition> sysdef)
    :IntegrationMethodTwoStep(sysdef,make_shared(ParticleGroup()))
    {
    //
    }
// TODO: using empty group mean no checking about same integrator on "particle", check another way for alchMD

// Overwrite Particle Group Methods with Alchemical Group Methods
unsigned int AlchemostatTwoStep::getNDOF()
    {
    return m_alchParticles->;
    }

unsigned int AlchemostatTwoStep::getRotationalNDOF()
    {
    m_exec_conf->msg->warning() << "Something is trying to use anisotropic DOF with AlchMD."
    return 0
    }

void AlchemostatTwoStep::validateGroup()
    {
    }

void AlchemostatTwoStep::randomizeVelocities(unsigned int timestep)
    {
    m_exec_conf->msg->warning() << "AlchMD hasn't implemented randomized velocities."
    }

// TODO: initializeIntegratorVariables? Make a standard one for defaults, easy to add to?
// nTimestep = 1 would be reasonable

// TODO: exports
