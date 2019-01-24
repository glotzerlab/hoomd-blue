// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/SystemDefinition.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Profiler.h"

#include <memory>

#ifndef __INTEGRATION_METHOD_TWO_STEP_H__
#define __INTEGRATION_METHOD_TWO_STEP_H__

#ifdef ENABLE_MPI
//! Forward declaration
class Communicator;
#endif

/*! \file IntegrationMethodTwoStep.h
    \brief Declares a base class for all two-step integration methods
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps
/*! \b Overview
    A large class of integrators can be implemented in two steps:
    - Update position and velocity (w/ current accel)
    - Sum accelerations at the current position
    - Update position and velocity again (w/ newly calculated accel)

    It is also sometimes desirable to run part of the system with one integration method (i.e. NVE) and part with
    another (NPT). Or part of the system with an integrator and the other part none at all. To facilitate this, the
    IntegrationMethodTwoStep is being created. It is a generic class, of which sub classes (TwoStepNVT, TwoStepNVE, ...)
    will implement the specific two step integration method. A single integrator, IntegratorTwoStep, can contain
    several two step integration methods. It calls the first step on all of them, then calculates the forces, and then
    calls the second step on all methods. In this way, the entire system will be integrated forward correctly.

    This design is chosen so that a single integration method is only applied to a group of particles. To enforce this
    design constraint, the group is specified in the constructor to the base class method.

    However, some care needs to be put into thinking about the computation of the net force / accelerations. Prior to
    implementing IntegrationMethodTwoStep, Integrators on the CPU have the net force and acceleration summed in
    Integrator::computeAccelerations. While the GPU ones only compute the forces in that call, and sum the net force
    and acceleration within the 2nd step of the integrator. In an interaction with groups, this is not going to work
    out. If one integration method worked one way and another worked the other in the same IntegratorTwoStep, then
    the net force / acceleration would probably not be calculated properly. To avoid this problem, a net force and
    virial will summed within Integrator::computeNetForce() / Integrator::computeNetForceGPU() which is called at the
    proper time in IntegratorTwoStep() (and presumably other integration routines).

    One small note: each IntegrationTwoStep will have a deltaT. The value of this will be set by the integrator when
    Integrator::setDeltaT is called to ensure that all integration methods have the same delta t set.

    <b>Integrator variables</b>

    Integrator variables are registered and tracked, if needed, through the IntegratorData interface. Because of the
    need for valid restart tracking (see below), \b all integrators register even if they do not need to save state
    information.

    Furthermore, the base class IntegratorTwoStep needs to know whether or not it should recalculate the "first step"
    accelerations. Accelerations are saved in the restart file, so if a restart is valid for all of the integration
    methods, it should skip that step. To facilitate this, derived classes should call setValidRestart(true) if they
    have valid restart information.

    <b>Thermodynamic properties</b>

    Thermodynamic properties on given groups are computed by ComputeThermo. See the documentation of ComputeThermo for
    its design and logging capabilities. To compute temperature properly, ComputeThermo needs the number of degrees of
    freedom. Only the Integrator can know that as it is the integrator that grants degrees of freedom to the particles.
    hoomd will break the dependency requirement. At the start of every run, hoomd will ask for an updated
    NDOF for every ComputeThermo group and set it.

    For IntegratorTwoStep, each IntegrationMethodTwoStep will compute its own contribution to the degrees of freedom
    for each particle in the group. IntegratorTwoStep will sum the contributions to get the total. At that point,
    D will be deducted from the total to get the COM motion constraint correct.

    <b>Design requirements</b>
    Due to the nature of allowing multiple integration methods to run at once, some strict guidelines need to be laid
    down.
    -# All methods must use the same \a deltaT (this is enforced by having IntegratorTwoStep call setDeltaT on all of
       the methods inside it.
    -# integrateStepOne uses the particle acceleration currently set to advance particle positions forward one full
       step, and velocities are advanced forward a half step.
    -# integrateStepTwo assigns the current particle acceleration from the net force and updates the velocities
       forward for the second half step
    -# each integration method only applies these operations to the particles contained within its group (exceptions
       are allowed when box rescaling is needed)

    <b>Design items still left to do:</b>

    Interaction with logger: perhaps the integrator should forward log value queries on to the integration method?
    each method could be given a user name so that they are logged in user-controlled columns. This provides a window
    into the internally computed state variables logging per method.

    \ingroup updaters
*/
class PYBIND11_EXPORT IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        IntegrationMethodTwoStep(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group);
        virtual ~IntegrationMethodTwoStep() {}

        //! Abstract method that performs the first step of the integration
        /*! \param timestep Current time step
        */
        virtual void integrateStepOne(unsigned int timestep) {}

        //! Abstract method that performs the second step of the integration
        /*! \param timestep Current time step
        */
        virtual void integrateStepTwo(unsigned int timestep)
            {
            }

        //! Sets the profiler for the integration method to use
        void setProfiler(std::shared_ptr<Profiler> prof);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            }

        //! Returns a list of log quantities this compute calculates
        /*! The base class implementation just returns an empty vector. Derived classes should override
            this behavior and return a list of quantities that they log.

            See Logger for more information on what this is about.
        */
        virtual std::vector< std::string > getProvidedLogQuantities()
            {
            return std::vector< std::string >();
            }

        //! Calculates the requested log value and returns it
        /*! \param quantity Name of the log quantity to get
            \param timestep Current time step of the simulation
            \param my_quantity_flag Returns true if this method tracks this quantity

            The base class just returns 0. Derived classes should override this behavior and return
            the calculated value for the given quantity. Only quantities listed in
            the return value getProvidedLogQuantities() will be requested from
            getLogValue().

            See Logger for more information on what this is about.
        */
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep,  bool &my_quantity_flag)
            {
            return Scalar(0.0);
            }

        //! Change the timestep
        void setDeltaT(Scalar deltaT);

        //! Access the group
        std::shared_ptr<ParticleGroup> getGroup() { return m_group; }

        //! Get whether this restart was valid
        bool isValidRestart() { return m_valid_restart; }

        //! Get the number of degrees of freedom granted to a given group
        virtual unsigned int getNDOF(std::shared_ptr<ParticleGroup> query_group);

        //! Get needed pdata flags
        /*! Not all fields in ParticleData are computed by default. When derived classes need one of these optional
            fields, they must return the requested fields in getRequestedPDataFlags().
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            return PDataFlags(0);
            }

        //! Validate that all members in the particle group are valid (throw an exception if they are not)
        virtual void validateGroup();

#ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm MPI communication class
         */
        void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            assert(comm);
            m_comm = comm;
            }
#endif

        //! Set (an-)isotropic integration mode
        /*! \param aniso True if anisotropic integration is requested
         */
        void setAnisotropic(bool aniso)
            {
            // warn if we are moving isotropic->anisotropic and we
            // find no rotational degrees of freedom
            if (!m_aniso && aniso && !this->getRotationalNDOF(m_group))
                {
                    m_exec_conf->msg->warning() << "Integrator #"<<  m_integrator_id <<
                        ": Anisotropic integration requested, but no rotational "
                        "degrees of freedom found for its group" << std::endl;
                }

            m_aniso = aniso;
            }

        //! Return if we are integrating anisotropically
        bool getAnisotropic() const { return m_aniso; }

        //! Compute rotational degrees of freedom
        /*! \param query_group The group of particles to compute rotational DOF for
         */
        virtual unsigned int getRotationalNDOF(std::shared_ptr<ParticleGroup> query_group);

        void setRandomizeVelocitiesParams(Scalar T_randomize, unsigned int seed_randomize)
            {
            m_T_randomize = T_randomize;
            m_seed_randomize = seed_randomize;
            m_shouldRandomize = true;
            }

        virtual void randomizeVelocities(unsigned int timestep);

        //! Reinitialize the integration variables if needed (implemented in the actual subclasses)
        virtual void initializeIntegratorVariables() {}

    protected:
        const std::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this method is associated with
        const std::shared_ptr<ParticleGroup> m_group;     //!< The group of particles this method works on
        const std::shared_ptr<ParticleData> m_pdata;      //!< The particle data this method is associated with
        std::shared_ptr<Profiler> m_prof;                 //!< The profiler this method is to use
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        bool m_aniso;                                       //!< True if anisotropic integration is requested

        /*! Member variables for randomizeVelocities(). */
        Scalar m_T_randomize = 0;
        unsigned int m_seed_randomize = 0;
        bool m_shouldRandomize = false;

        Scalar m_deltaT;                                    //!< The time step

        //! helper function to get the integrator variables from the particle data
        const IntegratorVariables& getIntegratorVariables()
            {
            return m_sysdef->getIntegratorData()->getIntegratorVariables(m_integrator_id);
            }

        //! helper function to store the integrator variables in the particle data
        void setIntegratorVariables(const IntegratorVariables& variables)
            {
            m_sysdef->getIntegratorData()->setIntegratorVariables(m_integrator_id, variables);
            }

        //! helper function to check if the restart information (if applicable) is usable
        bool restartInfoTestValid(const IntegratorVariables& v, std::string type, unsigned int nvariables);

        //! Set whether this restart is valid
        void setValidRestart(bool b) { m_valid_restart = b; }

#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;             //!< The communicator to use for MPI
#endif
    private:
        unsigned int m_integrator_id;                       //!< Registered integrator id to access the state variables
        bool m_valid_restart;                               //!< True if the restart info was valid when loading
    };

//! Exports the IntegrationMethodTwoStep class to python
void export_IntegrationMethodTwoStep(pybind11::module& m);

#endif // #ifndef __INTEGRATION_METHOD_TWO_STEP_H__
