// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file IntegratorTwoStep.cc
    \brief Defines the IntegratorTwoStep class
*/


#include "IntegratorTwoStep.h"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

IntegratorTwoStep::IntegratorTwoStep(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : Integrator(sysdef, deltaT), m_prepared(false), m_gave_warning(false),
    m_aniso_mode(Automatic)
    {
    m_exec_conf->msg->notice(5) << "Constructing IntegratorTwoStep" << endl;
    }

IntegratorTwoStep::~IntegratorTwoStep()
    {
    m_exec_conf->msg->notice(5) << "Destroying IntegratorTwoStep" << endl;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        m_comm->getComputeCallbackSignal().disconnect<IntegratorTwoStep, &IntegratorTwoStep::updateRigidBodies>(this);
        }
    #endif
    }

/*! \param prof The profiler to set
    Sets the profiler both for this class and all of the contained integration methods
*/
void IntegratorTwoStep::setProfiler(std::shared_ptr<Profiler> prof)
    {
    Integrator::setProfiler(prof);

    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->setProfiler(prof);
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > IntegratorTwoStep::getProvidedLogQuantities()
    {
    std::vector<std::string> combined_result;
    std::vector<std::string> result;

    // Get base class provided log quantities
    result = Integrator::getProvidedLogQuantities();
    combined_result.insert(combined_result.end(), result.begin(), result.end());

    // add integrationmethod quantities
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        result = (*method)->getProvidedLogQuantities();
        combined_result.insert(combined_result.end(), result.begin(), result.end());
        }
    return combined_result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
*/
Scalar IntegratorTwoStep::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    bool quantity_flag = false;
    Scalar log_value;

    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        log_value = (*method)->getLogValue(quantity,timestep,quantity_flag);
        if (quantity_flag) return log_value;
        }
    return Integrator::getLogValue(quantity, timestep);
    }

/*! \param timestep Current time step of the simulation
    \post All integration methods previously added with addIntegrationMethod() are applied in order to move the system
          state variables forward to \a timestep+1.
    \post Internally, all forces added via Integrator::addForceCompute are evaluated at \a timestep+1
*/
void IntegratorTwoStep::update(unsigned int timestep)
    {
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0)
        {
        m_exec_conf->msg->warning() << "integrate.mode_standard: No integration methods are set, continuing anyways." << endl;
        m_gave_warning = true;
        }

    // ensure that prepRun() has been called
    assert(m_prepared);

    if (m_prof)
        m_prof->push("Integrate");

    // perform the first step of the integration on all groups
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);

    if (m_prof)
        m_prof->pop();

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time step

        // also updates rigid bodies after ghost updating
        m_comm->communicate(timestep+1);
        }
    else
#endif
        {
        updateRigidBodies(timestep+1);
        }

    // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    // Call HalfStep hook
    if (m_half_step_hook)
        {
        m_half_step_hook->update(timestep+1);
        }

    if (m_prof)
        m_prof->push("Integrate");

    // perform the second step of the integration on all groups
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);

    /* NOTE: For composite particles, it is assumed that positions and orientations are not updated
       in the second step.

       Otherwise we would have to update ghost positions for central particles
       here in order to update the constituent particles.

       TODO: check this assumptions holds for all integrators
     */

    if (m_prof)
        m_prof->pop();
    }

/*! \param deltaT new deltaT to set
    \post \a deltaT is also set on all contained integration methods
*/
void IntegratorTwoStep::setDeltaT(Scalar deltaT)
    {
    Integrator::setDeltaT(deltaT);

    // set deltaT on all methods already added
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->setDeltaT(deltaT);
    }

/*! \param new_method New integration method to add to the integrator
    Before the method is added, it is checked to see if the group intersects with any of the groups integrated by
    existing methods. If an intersection is found, an error is issued. If no intersection is found, setDeltaT
    is called on the method and it is added to the list.
*/
void IntegratorTwoStep::addIntegrationMethod(std::shared_ptr<IntegrationMethodTwoStep> new_method)
    {
    // check for intersections with existing methods
    std::shared_ptr<ParticleGroup> new_group = new_method->getGroup();

    if (new_group->getNumMembersGlobal() == 0)
        m_exec_conf->msg->warning() << "integrate.mode_standard: An integration method has been added that operates on zero particles." << endl;

    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
        std::shared_ptr<ParticleGroup> intersection = ParticleGroup::groupIntersection(new_group, current_group);

        if (intersection->getNumMembersGlobal() > 0)
            {
            m_exec_conf->msg->error() << "integrate.mode_standard: Multiple integration methods are applied to the same particle" << endl;
            throw std::runtime_error("Error adding integration method");
            }
        }

    // ensure that the method has a matching deltaT
    new_method->setDeltaT(m_deltaT);

    // add it to the list
    m_methods.push_back(new_method);
    }

/*! \post All integration methods are removed from this integrator
*/
void IntegratorTwoStep::removeAllIntegrationMethods()
    {
    m_methods.clear();
    m_gave_warning = false;
    }

/*! \param fc ForceComposite to add
*/
void IntegratorTwoStep::addForceComposite(std::shared_ptr<ForceComposite> fc)
    {
    assert(fc);
    m_composite_forces.push_back(fc);
    }

/*! Call removeForceComputes() to completely wipe out the list of force computes
    that the integrator uses to sum forces.
*/
void IntegratorTwoStep::removeForceComputes()
    {
    Integrator::removeForceComputes();

    // Remove ForceComposite objects
    m_composite_forces.clear();
    }


/*! \returns true If all added integration methods have valid restart information
*/
bool IntegratorTwoStep::isValidRestart()
    {
    bool res = true;

    // loop through all methods
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // and them all together
        res = res && (*method)->isValidRestart();
        }
    return res;
    }

/*! \returns true If all added integration methods have valid restart information
*/
void IntegratorTwoStep::initializeIntegrationMethods()
    {
    // loop through all methods
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // initialize each of them
        (*method)->initializeIntegratorVariables();
        }
    }

/*! \param group Group over which to count degrees of freedom.
    IntegratorTwoStep totals up the degrees of freedom that each integration method provide to the group.
    Three degrees of freedom are subtracted from the total to account for the constrained position of the system center of
    mass.
*/
unsigned int IntegratorTwoStep::getNDOF(std::shared_ptr<ParticleGroup> group)
    {
    int res = 0;

    // loop through all methods
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // dd them all together
        res += (*method)->getNDOF(group);
        }

    return res - m_sysdef->getNDimensions() - getNDOFRemoved();
    }

/*! \param group Group over which to count degrees of freedom.
    IntegratorTwoStep totals up the rotational degrees of freedom that each integration method provide to the group.
*/
unsigned int IntegratorTwoStep::getRotationalNDOF(std::shared_ptr<ParticleGroup> group)
    {
    int res = 0;

    bool aniso = false;

    // This is called before prepRun, so we need to determine the anisotropic modes independently here.
    // It cannot be done earlier, as the integration methods were not in place.
    // set (an-)isotropic integration mode
    switch (m_aniso_mode)
        {
        case Anisotropic:
            aniso = true;
            break;
        case Automatic:
        default:
            aniso = getAnisotropic();
            break;
        }

    m_exec_conf->msg->notice(8) << "IntegratorTwoStep: Setting anisotropic mode = " << aniso << std::endl;

    if (aniso)
        {
        // loop through all methods
        std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
        for (method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            // dd them all together
            res += (*method)->getRotationalNDOF(group);
            }
        }

    return res;
    }

/*!  \param mode Anisotropic integration mode to set
     Set the anisotropic integration mode
*/
void IntegratorTwoStep::setAnisotropicMode(AnisotropicMode mode)
    {
    m_aniso_mode = mode;
    }

/*! Compute accelerations if needed for the first step.
    If acceleration is available in the restart file, then just call computeNetForce so that net_force and net_virial
    are available for the logger. This solves ticket #393
*/
void IntegratorTwoStep::prepRun(unsigned int timestep)
    {
    bool aniso = false;

    // set (an-)isotropic integration mode
    switch (m_aniso_mode)
        {
        case Anisotropic:
            aniso = true;
            if(!getAnisotropic())
                m_exec_conf->msg->warning() << "Forcing anisotropic integration mode"
                    " with no forces coupling to orientation" << endl;
            break;
        case Isotropic:
            if(getAnisotropic())
                m_exec_conf->msg->warning() << "Forcing isotropic integration mode"
                    " with anisotropic forces defined" << endl;
            break;
        case Automatic:
        default:
            aniso = getAnisotropic();
            break;
        }

    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->setAnisotropic(aniso);

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // force particle migration and ghost exchange
        m_comm->forceMigrate();

        // perform communication
        m_comm->communicate(timestep);
        }
    else
#endif
        {
        updateRigidBodies(timestep);
        }

        // compute the net force on all particles
#ifdef ENABLE_CUDA
    if (m_exec_conf->exec_mode == ExecutionConfiguration::GPU)
        computeNetForceGPU(timestep);
    else
#endif
        computeNetForce(timestep);

    // accelerations only need to be calculated if the accelerations have not yet been set
    if (!m_pdata->isAccelSet())
        {
        computeAccelerations(timestep);
        m_pdata->notifyAccelSet();
        }

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->randomizeVelocities(timestep);

    m_prepared = true;
    }

/*! Return the combined flags of all integration methods.
*/
PDataFlags IntegratorTwoStep::getRequestedPDataFlags()
    {
    PDataFlags flags;

    // loop through all methods
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        // or them all together
        flags |= (*method)->getRequestedPDataFlags();
        }

    return flags;
    }

#ifdef ENABLE_MPI
//! Set the communicator to use
void IntegratorTwoStep::setCommunicator(std::shared_ptr<Communicator> comm)
    {
    // set Communicator in all methods
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
            (*method)->setCommunicator(comm);

    if (comm && !m_comm)
        {
        // on the first time setting the Communicator, connect our compute callback
        comm->getComputeCallbackSignal().connect<IntegratorTwoStep, &IntegratorTwoStep::updateRigidBodies>(this);
        }

    Integrator::setCommunicator(comm);
    }
#endif

//! Updates the rigid body constituent particles
void IntegratorTwoStep::updateRigidBodies(unsigned int timestep)
    {
    // slave any constituents of local composite particles
    for (auto force_composite = m_composite_forces.begin(); force_composite != m_composite_forces.end(); ++force_composite)
        (*force_composite)->updateCompositeParticles(timestep);
    }

/*! \param enable Enable/disable autotuning
    \param period period (approximate) in time steps when returning occurs
*/
void IntegratorTwoStep::setAutotunerParams(bool enable, unsigned int period)
    {
    Integrator::setAutotunerParams(enable, period);
    // set params in all methods
    std::vector< std::shared_ptr<IntegrationMethodTwoStep> >::iterator method;
    for (method = m_methods.begin(); method != m_methods.end(); ++method)
            (*method)->setAutotunerParams(enable, period);
    }

void export_IntegratorTwoStep(py::module& m)
    {
    py::class_<IntegratorTwoStep, std::shared_ptr<IntegratorTwoStep> >(m, "IntegratorTwoStep", py::base<Integrator>())
        .def(py::init< std::shared_ptr<SystemDefinition>, Scalar >())
        .def("addIntegrationMethod", &IntegratorTwoStep::addIntegrationMethod)
        .def("removeAllIntegrationMethods", &IntegratorTwoStep::removeAllIntegrationMethods)
        .def("setAnisotropicMode", &IntegratorTwoStep::setAnisotropicMode)
        .def("addForceComposite", &IntegratorTwoStep::addForceComposite)
        .def("removeForceComputes", &IntegratorTwoStep::removeForceComputes)
        .def("initializeIntegrationMethods", &IntegratorTwoStep::initializeIntegrationMethods)
        ;

    py::enum_<IntegratorTwoStep::AnisotropicMode>(m,"IntegratorAnisotropicMode")
        .value("Automatic", IntegratorTwoStep::AnisotropicMode::Automatic)
        .value("Anisotropic", IntegratorTwoStep::AnisotropicMode::Anisotropic)
        .value("Isotropic", IntegratorTwoStep::AnisotropicMode::Isotropic)
        .export_values()
        ;

    }
