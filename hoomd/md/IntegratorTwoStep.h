// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Integrator.h"

#include "ForceComposite.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/// Integrates the system forward one step with possibly multiple methods
/** See IntegrationMethodTwoStep for most of the design notes regarding group integration.
   IntegratorTwoStep merely implements most of the things discussed there.

    Notable design elements:
    - setDeltaT results in deltaT being set on all current integration methods
    - to interface with the python script, the m_methods vectors is exposed with a list like API.

   TODO: ensure that the user does not make a mistake and specify more than one method operating on
   a single particle

    There is a special registration mechanism for ForceComposites which run after the integration
   steps one and two, and which can use the updated particle positions and velocities to update any
   slaved degrees of freedom (rigid bodies).

    \ingroup updaters
*/
class PYBIND11_EXPORT IntegratorTwoStep : public Integrator
    {
    public:
    /// Constructor
    IntegratorTwoStep(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

    /// Destructor
    virtual ~IntegratorTwoStep();

    /// Take one timestep forward
    virtual void update(uint64_t timestep);

    /// Change the timestep
    virtual void setDeltaT(Scalar deltaT);

    /// Get the list of integration methods
    std::vector<std::shared_ptr<IntegrationMethodTwoStep>>& getIntegrationMethods()
        {
        return m_methods;
        }

    /// Get the number of degrees of freedom granted to a given group
    virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group);

    /// Get the number of degrees of freedom granted to a given group
    virtual Scalar getRotationalDOF(std::shared_ptr<ParticleGroup> group);

    /// Set the integrate orientation flag
    virtual void setIntegrateRotationalDOF(bool integrate_rotational_dofs);

    /// Set the integrate orientation flag
    virtual const bool getIntegrateRotationalDOF();

    /// Prepare for the run
    virtual void prepRun(uint64_t timestep);

    /// Get needed pdata flags
    virtual PDataFlags getRequestedPDataFlags();

    /// helper function to compute net force/virial
    virtual void computeNetForce(uint64_t timestep);

#ifdef ENABLE_HIP
    /// helper function to compute net force/virial on the GPU
    virtual void computeNetForceGPU(uint64_t timestep);
#endif

#ifdef ENABLE_MPI
    /// helper function to determine the ghost communication flags
    virtual CommFlags determineFlags(uint64_t timestep);
#endif

    /// Check if any forces introduce anisotropic degrees of freedom
    virtual bool areForcesAnisotropic();

    /// Updates the rigid body constituent particles
    virtual void updateRigidBodies(uint64_t timestep);

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning();

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete();

    /// Getter and setter for accessing rigid body objects in Python
    std::shared_ptr<ForceComposite> getRigid()
        {
        return m_rigid_bodies;
        }

    void setRigid(std::shared_ptr<ForceComposite> new_rigid)
        {
        m_rigid_bodies = new_rigid;
        }

    /// Validate method groups.
    void validateGroups();

    protected:
    std::vector<std::shared_ptr<IntegrationMethodTwoStep>>
        m_methods; //!< List of all the integration methods

    std::shared_ptr<ForceComposite> m_rigid_bodies; /// definition and updater for rigid bodies

    bool m_prepared; //!< True if preprun has been called

    /// True when orientation degrees of freedom should be integrated
    bool m_integrate_rotational_dof = false;
    };

    } // end namespace md
    } // end namespace hoomd
