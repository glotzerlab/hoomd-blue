// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file IntegratorData.cc
    \brief Contains all code for IntegratorData.
 */

#include "IntegratorData.h"

unsigned int IntegratorData::registerIntegrator()
    {
    // assign the next available slot
    unsigned int i = m_num_registered;
    m_num_registered++;

    // grow the vector if it needs to be
    if (i >= m_integrator_variables.size())
        {
        m_integrator_variables.resize(i+1);
        }

    // return the handle
    return i;
    }
