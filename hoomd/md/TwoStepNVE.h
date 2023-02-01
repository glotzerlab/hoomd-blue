// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"

#ifndef __TWO_STEP_NVE_H__
#define __TWO_STEP_NVE_H__

/*! \file TwoStepNVE.h
    \brief Declares the TwoStepNVE class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepNVE : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepNVE(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group);
    virtual ~TwoStepNVE();

    /// Get the movement limit
    std::shared_ptr<Variant> getLimit();

    //! Sets the movement limit
    void setLimit(std::shared_ptr<Variant>& limit);

    /// Get zero force
    bool getZeroForce();

    //! Sets the zero force option
    /*! \param zero_force Set to true to specify that the integration with a zero net force on each
       of the particles in the group
    */
    void setZeroForce(bool zero_force);

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    protected:
    //!< The maximum distance a particle is to move in one step
    std::shared_ptr<Variant> m_limit;
    bool m_zero_force; //!< True if the integration step should ignore computed forces
    };

    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_NVE_H__
