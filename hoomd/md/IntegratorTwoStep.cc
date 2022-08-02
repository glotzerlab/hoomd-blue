// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegratorTwoStep.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::md::IntegrationMethodTwoStep>>);

using namespace std;

namespace hoomd
    {
namespace md
    {
IntegratorTwoStep::IntegratorTwoStep(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : Integrator(sysdef, deltaT), m_prepared(false), m_gave_warning(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing IntegratorTwoStep" << endl;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getComputeCallbackSignal()
            .connect<IntegratorTwoStep, &IntegratorTwoStep::updateRigidBodies>(this);
        }
#endif
    }

IntegratorTwoStep::~IntegratorTwoStep()
    {
    m_exec_conf->msg->notice(5) << "Destroying IntegratorTwoStep" << endl;

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getComputeCallbackSignal()
            .disconnect<IntegratorTwoStep, &IntegratorTwoStep::updateRigidBodies>(this);
        }
#endif
    }

/*! \param timestep Current time step of the simulation
    \post All integration methods in m_methods are applied in order to move the system state
    variables forward to \a timestep+1.
    \post Internally, all forces present in the m_forces std::vector are evaluated at \a timestep+1
*/
void IntegratorTwoStep::update(uint64_t timestep)
    {
    Integrator::update(timestep);

    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0)
        {
        m_exec_conf->msg->warning() << "MD Integrator has no integration methods." << endl;
        m_gave_warning = true;
        }

    // ensure that prepRun() has been called
    assert(m_prepared);

    // perform the first step of the integration on all groups
    for (auto& method : m_methods)
        {
        // deltaT should probably be passed as an argument, but that would require modifying many
        // files. Work around this by calling setDeltaT every timestep.
        method->setAnisotropic(m_integrate_rotational_dof);
        method->setDeltaT(m_deltaT);
        method->integrateStepOne(timestep);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time
        // step

        // also updates rigid bodies after ghost updating
        m_comm->communicate(timestep + 1);
        }
    else
#endif
        {
        updateRigidBodies(timestep + 1);
        }

    // compute the net force on all particles
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        computeNetForceGPU(timestep + 1);
    else
#endif
        computeNetForce(timestep + 1);

    // Call HalfStep hook
    if (m_half_step_hook)
        {
        m_half_step_hook->update(timestep + 1);
        }

    // perform the second step of the integration on all groups
    // reversed for integrators so that the half steps will be performed symmetrically
    for (auto method_ptr = m_methods.rbegin(); method_ptr != m_methods.rend(); method_ptr++)
        {
        auto method = *method_ptr;
        method->integrateStepTwo(timestep);
        method->includeRATTLEForce(timestep + 1);
        }

    /* NOTE: For composite particles, it is assumed that positions and orientations are not updated
       in the second step.

       Otherwise we would have to update ghost positions for central particles
       here in order to update the constituent particles.

       TODO: check this assumptions holds for all integrators
     */
    }

/*! \param deltaT new deltaT to set
    \post \a deltaT is also set on all contained integration methods
*/
void IntegratorTwoStep::setDeltaT(Scalar deltaT)
    {
    Integrator::setDeltaT(deltaT);

    // set deltaT on all methods already added
    for (auto& method : m_methods)
        {
        method->setDeltaT(deltaT);
        }
    if (m_rigid_bodies)
        {
        m_rigid_bodies->setDeltaT(deltaT);
        }
    }

/*! \param group Group over which to count degrees of freedom.

    IntegratorTwoStep totals up the degrees of freedom that each integration method provide to the
    group.

    When the user has only one momentum conserving integration method applied to the all group,
    getNDOF subtracts n_dimensions degrees of freedom from the system to account for the pinned
    center of mass. When the query group is not the group of all particles, spread these these
    removed DOF proportionately so that the results given by one ComputeThermo on the all group are
    consitent with the average of many ComputeThermo's on disjoint subset groups.
*/
Scalar IntegratorTwoStep::getTranslationalDOF(std::shared_ptr<ParticleGroup> group)
    {
    // proportionately remove n_dimensions DOF when there is only one momentum conserving
    // integration method
    Scalar periodic_dof_removed = 0;
    if (group->getNumMembersGlobal() == m_pdata->getNGlobal() && m_methods.size() == 1
        && m_methods[0]->isMomentumConserving())
        {
        periodic_dof_removed
            = Scalar(m_sysdef->getNDimensions())
              * (Scalar(group->getNumMembersGlobal()) / Scalar(m_pdata->getNGlobal()));
        }

    // loop through all methods and add up the number of DOF They apply to the group
    Scalar total = 0;
    for (auto& method : m_methods)
        {
        total += method->getTranslationalDOF(group);
        }

    return total - periodic_dof_removed - getNDOFRemoved(group);
    }

/*! \param group Group over which to count degrees of freedom.
    IntegratorTwoStep totals up the rotational degrees of freedom that each integration method
   provide to the group.
*/
Scalar IntegratorTwoStep::getRotationalDOF(std::shared_ptr<ParticleGroup> group)
    {
    double res = 0;

    if (m_integrate_rotational_dof)
        {
        for (auto& method : m_methods)
            {
            res += method->getRotationalDOF(group);
            }
        }

    return res;
    }

/*!  \param integrate_rotational_dofs true to integrate orientations, false to not
 */
void IntegratorTwoStep::setIntegrateRotationalDOF(bool integrate_rotational_dof)
    {
    m_integrate_rotational_dof = integrate_rotational_dof;
    }

const bool IntegratorTwoStep::getIntegrateRotationalDOF()
    {
    return m_integrate_rotational_dof;
    }

/*! Compute accelerations if needed for the first step.
    If acceleration is available in the restart file, then just call computeNetForce so that
    net_force and net_virial are available in Python. This solves ticket #393
*/
void IntegratorTwoStep::prepRun(uint64_t timestep)
    {
    Integrator::prepRun(timestep);
    if (m_integrate_rotational_dof && !areForcesAnisotropic())
        {
        m_exec_conf->msg->warning() << "Requested integration of orientations, but no forces"
                                       " provide torques."
                                    << endl;
        }
    if (!m_integrate_rotational_dof && areForcesAnisotropic())
        {
        m_exec_conf->msg->warning() << "Forces provide torques, but integrate_rotational_dof is"
                                       "false."
                                    << endl;
        }

    for (auto& method : m_methods)
        method->setAnisotropic(m_integrate_rotational_dof);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // force particle migration and ghost exchange
        m_comm->forceMigrate();

        // perform communication
        m_comm->communicate(timestep);
        }
    else
#endif
        if (m_rigid_bodies)
        {
        m_rigid_bodies->validateRigidBodies();
        updateRigidBodies(timestep);
        }

    // compute the net force on all particles
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
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

    for (auto& method : m_methods)
        method->includeRATTLEForce(timestep);

    m_prepared = true;
    }

/*! Return the combined flags of all integration methods.
 */
PDataFlags IntegratorTwoStep::getRequestedPDataFlags()
    {
    PDataFlags flags;

    // loop through all methods
    for (auto& method : m_methods)
        {
        // or them all together
        flags |= method->getRequestedPDataFlags();
        }

    return flags;
    }

//! Updates the rigid body constituent particles
void IntegratorTwoStep::updateRigidBodies(uint64_t timestep)
    {
    // update the composite particle positions of any rigid bodies
    if (m_rigid_bodies)
        {
        m_rigid_bodies->updateCompositeParticles(timestep);
        }
    }

/*! \param enable Enable/disable autotuning
    \param period period (approximate) in time steps when returning occurs
*/
void IntegratorTwoStep::setAutotunerParams(bool enable, unsigned int period)
    {
    Integrator::setAutotunerParams(enable, period);
    // set params in all methods
    for (auto& method : m_methods)
        method->setAutotunerParams(enable, period);
    }

/// helper function to compute net force/virial
void IntegratorTwoStep::computeNetForce(uint64_t timestep)
    {
    if (m_rigid_bodies)
        {
        m_rigid_bodies->validateRigidBodies();
        m_constraint_forces.push_back(m_rigid_bodies);
        }
    Integrator::computeNetForce(timestep);
    if (m_rigid_bodies)
        {
        m_constraint_forces.pop_back();
        }
    }

#ifdef ENABLE_HIP
/// helper function to compute net force/virial on the GPU
void IntegratorTwoStep::computeNetForceGPU(uint64_t timestep)
    {
    if (m_rigid_bodies)
        {
        m_rigid_bodies->validateRigidBodies();
        m_constraint_forces.push_back(m_rigid_bodies);
        }
    Integrator::computeNetForceGPU(timestep);
    if (m_rigid_bodies)
        {
        m_constraint_forces.pop_back();
        }
    }
#endif

#ifdef ENABLE_MPI
/// helper function to determine the ghost communication flags
CommFlags IntegratorTwoStep::determineFlags(uint64_t timestep)
    {
    auto flags = Integrator::determineFlags(timestep);
    if (m_rigid_bodies)
        {
        flags |= m_rigid_bodies->getRequestedCommFlags(timestep);
        }
    return flags;
    }
#endif

/// Check if any forces introduce anisotropic degrees of freedom
bool IntegratorTwoStep::areForcesAnisotropic()
    {
    auto is_anisotropic = Integrator::areForcesAnisotropic();
    if (m_rigid_bodies)
        {
        is_anisotropic |= m_rigid_bodies->isAnisotropic();
        }
    return is_anisotropic;
    }

namespace detail
    {
void export_IntegratorTwoStep(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<IntegrationMethodTwoStep>>>(
        m,
        "IntegrationMethodList");

    pybind11::class_<IntegratorTwoStep, Integrator, std::shared_ptr<IntegratorTwoStep>>(
        m,
        "IntegratorTwoStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property_readonly("methods", &IntegratorTwoStep::getIntegrationMethods)
        .def_property("rigid", &IntegratorTwoStep::getRigid, &IntegratorTwoStep::setRigid)
        .def_property("integrate_rotational_dof",
                      &IntegratorTwoStep::getIntegrateRotationalDOF,
                      &IntegratorTwoStep::setIntegrateRotationalDOF)
        .def_property("half_step_hook",
                      nullptr,
                      [](IntegratorTwoStep& self, std::shared_ptr<HalfStepHook> hook)
                      {
                          if (hook)
                              self.setHalfStepHook(hook);
                          else // `if hook is None` on the python side
                              self.removeHalfStepHook();
                      });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
