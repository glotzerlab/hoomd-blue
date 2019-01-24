// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/Integrator.h"
#include "IntegrationMethodTwoStep.h"

#include "ForceComposite.h"

#ifndef __INTEGRATOR_TWO_STEP_H__
#define __INTEGRATOR_TWO_STEP_H__

/*! \file IntegratorTwoStep.h
    \brief Declares an integrator for performing two-step integration on multiple groups
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates the system forward one step with possibly multiple methods
/*! See IntegrationMethodTwoStep for most of the design notes regarding group integration. IntegratorTwoStep merely
    implements most of the things discussed there.

    Notable design elements:
    - setDeltaT results in deltaT being set on all current integration methods
    - to ensure that new methods also get set, addIntegrationMethod() also calls setDeltaT on the method
    - to interface with the python script, a removeAllIntegrationMethods() method is provided to clear the list so they
      can be cleared and re-added from hoomd's internal list

    To ensure that the user does not make a mistake and specify more than one method operating on a single particle,
    the particle groups are checked for intersections whenever a new method is added in addIntegrationMethod()

    There is a special registration mechanism for ForceComposites which run after the integration steps
    one and two, and which can use the updated particle positions and velocities to update any slaved degrees
    of freedom (rigid bodies).

    \ingroup updaters
*/
class PYBIND11_EXPORT IntegratorTwoStep : public Integrator
    {
    public:
        //! Anisotropic integration mode: Automatic (detect whether
        //! aniso forces are defined), Anisotropic (integrate
        //! rotational degrees of freedom regardless of whether
        //! anything is defining them), and Isotropic (don't integrate
        //! rotational degrees of freedom)
        enum AnisotropicMode {Automatic, Anisotropic, Isotropic};

        //! Constructor
        IntegratorTwoStep(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

        //! Destructor
        virtual ~IntegratorTwoStep();

        //! Sets the profiler for the compute to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof);

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);

        //! Add a new integration method to the list that will be run
        virtual void addIntegrationMethod(std::shared_ptr<IntegrationMethodTwoStep> new_method);

        //! Remove all integration methods
        virtual void removeAllIntegrationMethods();

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF(std::shared_ptr<ParticleGroup> group);

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getRotationalNDOF(std::shared_ptr<ParticleGroup> group);

        //! Set the anisotropic mode of the integrator
        virtual void setAnisotropicMode(AnisotropicMode mode);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

        //! Get needed pdata flags
        virtual PDataFlags getRequestedPDataFlags();

        //! Add a ForceComposite to the list
        virtual void addForceComposite(std::shared_ptr<ForceComposite> fc);

        //! Removes all ForceComputes from the list
        virtual void removeForceComputes();

#ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm The Communicator
         */
        virtual void setCommunicator(std::shared_ptr<Communicator> comm);
#endif

        //! Updates the rigid body constituent particles
        virtual void updateRigidBodies(unsigned int timestep);

        //! Set autotuner parameters
        virtual void setAutotunerParams(bool enable, unsigned int period);

        //! (Re-)initialize the integration method
        void initializeIntegrationMethods();

    protected:
        //! Helper method to test if all added methods have valid restart information
        bool isValidRestart();

        std::vector< std::shared_ptr<IntegrationMethodTwoStep> > m_methods;   //!< List of all the integration methods

        bool m_prepared;              //!< True if preprun has been called
        bool m_gave_warning;          //!< True if a warning has been given about no methods added
        AnisotropicMode m_aniso_mode; //!< Anisotropic mode for this integrator

        std::vector< std::shared_ptr<ForceComposite> > m_composite_forces; //!< A list of active composite forces
    };

//! Exports the IntegratorTwoStep class to python
void export_IntegratorTwoStep(pybind11::module& m);

#endif // #ifndef __INTEGRATOR_TWO_STEP_H__
