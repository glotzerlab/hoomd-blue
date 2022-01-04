// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file IntegratorData.cc
    \brief Contains all code for IntegratorData.
 */

#include "IntegratorData.h"

namespace hoomd
    {
unsigned int IntegratorData::registerIntegrator()
    {
    // assign the next available slot
    unsigned int i = m_num_registered;
    m_num_registered++;

    // grow the vector if it needs to be
    if (i >= m_integrator_variables.size())
        {
        m_integrator_variables.resize(i + 1);
        }

    // return the handle
    return i;
    }

    } // end namespace hoomd
