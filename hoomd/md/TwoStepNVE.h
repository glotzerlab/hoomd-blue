// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_NVE_H__
#define __TWO_STEP_NVE_H__

/*! \file TwoStepNVE.h
    \brief Declares the TwoStepNVE class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVE : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVE(std::shared_ptr<SystemDefinition> sysdef,
                   std::shared_ptr<ParticleGroup> group,
                   bool skip_restart=false);
        virtual ~TwoStepNVE();

        //! Sets the movement limit
        void setLimit(Scalar limit);

        //! Removes the limit
        void removeLimit();

        //! Sets the zero force option
        /*! \param zero_force Set to true to specify that the integration with a zero net force on each of the particles
                              in the group
        */
        void setZeroForce(bool zero_force)
            {
            m_zero_force = zero_force;
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        bool m_limit;       //!< True if we should limit the distance a particle moves in one step
        Scalar m_limit_val; //!< The maximum distance a particle is to move in one step
        bool m_zero_force;  //!< True if the integration step should ignore computed forces
    };

//! Exports the TwoStepNVE class to python
void export_TwoStepNVE(pybind11::module& m);

#endif // #ifndef __TWO_STEP_NVE_H__
