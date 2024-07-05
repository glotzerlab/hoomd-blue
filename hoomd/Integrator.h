// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#pragma once

#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "HalfStepHook.h"
#include "ParticleGroup.h"
#include "Updater.h"
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace hoomd
    {
/// Base class that defines an integrator
/** An Integrator steps the entire simulation forward one time step in time.
    Prior to calling update(timestep), the system is at time step \a timestep.
    After the call to update completes, the system is at \a timestep + 1.

    All integrators have the common property that they add up many forces to get the net force on
    each particle. This task is performed by the base class Integrator. Similarly, all integrators
    share the property that they have a time step, \a deltaT.

    Any number of ForceComputes can be used to specify the net force for use with this integrator.
    They are added through m_forces accessed via getForces. Any number of forces can be added in
    this way.

    All forces added to m_forces are computed independently and then totaled up to calculate the net
    force and energy on each particle. Constraint forces (ForceConstraint) are unique in that they
    need to be computed \b after the net forces is already available. To implement this behavior,
    add constraint forces to m_constraint_forces through getConstraintForces. All constraint forces
    will be computed independently and will be able to read the current unconstrained net force.
    The particles that separate constraint forces interact with should not overlap. Degrees of
    freedom removed via the constraint forces can be totaled up with a call to getNDOFRemoved for
    convenience in derived classes implementing correct counting in getTranslationalDOF() and
    getRotationalDOF().

    Integrators take "ownership" of the particle's accelerations. Any other updater that modifies
    the particles accelerations will produce undefined results. If accelerations are to be modified,
    they must be done through forces, and added to an Integrator via the m_forces std::vector.

    No such ownership is taken of the particle positions and velocities. Other Updaters
    can modify particle positions and velocities as they wish. Those updates will be taken
    into account by the Integrator. It would probably make the most sense to have such updaters
    run BEFORE the Integrator, since an Integrator actually moves the particles to the next time
    step. This is handled correctly by System.

    \ingroup updaters
*/
class PYBIND11_EXPORT Integrator : public Updater
    {
    public:
    /// Constructor
    Integrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

    /// Destructor
    virtual ~Integrator();

    /// Take one timestep forward
    virtual void update(uint64_t timestep);

    /// Get the list of force computes
    std::vector<std::shared_ptr<ForceCompute>>& getForces()
        {
        return m_forces;
        }

    /// Get the list of force computes
    std::vector<std::shared_ptr<ForceConstraint>>& getConstraintForces()
        {
        return m_constraint_forces;
        }

    /// Set the half step hook.
    virtual void setHalfStepHook(std::shared_ptr<HalfStepHook> hook)
        {
        m_half_step_hook = hook;
        }

    // Get the half step hook.
    virtual std::shared_ptr<HalfStepHook> getHalfStepHook()
        {
        return m_half_step_hook;
        }

    /// Change the timestep
    virtual void setDeltaT(Scalar deltaT);

    /// Return the timestep
    Scalar getDeltaT();

    /// Update the number of degrees of freedom for a group
    /** @param group Group to set the degrees of freedom for.
     */
    void updateGroupDOF(std::shared_ptr<ParticleGroup> group)
        {
        group->setTranslationalDOF(getTranslationalDOF(group));
        group->setRotationalDOF(getRotationalDOF(group));
        }

    /// Get the number of degrees of freedom granted to a given group
    /** @param group Group over which to count degrees of freedom.
        Base class Integrator returns 0. Derived classes should override.
    */
    virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group)
        {
        return 0;
        }

    /** Get the number of rotational degrees of freedom granted to a given group

        @param group Group over which to count degrees of freedom.
        Base class Integrator returns 0. Derived classes should override.
    */
    virtual Scalar getRotationalDOF(std::shared_ptr<ParticleGroup> group)
        {
        return 0;
        }

    /// Count the total number of degrees of freedom removed by all constraint forces
    Scalar getNDOFRemoved(std::shared_ptr<ParticleGroup> query);

    /// Compute the linear momentum of the system
    virtual vec3<double> computeLinearMomentum();

    /// Prepare for the run
    virtual void prepRun(uint64_t timestep);

#ifdef ENABLE_MPI
    /// Callback for pre-computing the forces
    void computeCallback(uint64_t timestep);
#endif

    /// Reset stats counters for children objects
    virtual void resetStats()
        {
        for (auto& force : m_forces)
            {
            force->resetStats();
            }

        for (auto& constraint_force : m_constraint_forces)
            {
            constraint_force->resetStats();
            }
        }

    /// Start autotuning kernel launch parameters
    virtual void startAutotuning()
        {
        Updater::startAutotuning();
        for (auto& force : m_forces)
            {
            force->startAutotuning();
            }
        }

    /// Check if autotuning is complete.
    virtual bool isAutotuningComplete()
        {
        bool result = Updater::isAutotuningComplete();
        for (auto& force : m_forces)
            {
            result = result && force->isAutotuningComplete();
            }
        return result;
        }

    protected:
    /// The step size
    Scalar m_deltaT;

    /// List of all the force computes
    std::vector<std::shared_ptr<ForceCompute>> m_forces;

    /// List of all the constraints
    std::vector<std::shared_ptr<ForceConstraint>> m_constraint_forces;

    /// The HalfStepHook, if active
    std::shared_ptr<HalfStepHook> m_half_step_hook;

    /// helper function to compute initial accelerations
    void computeAccelerations(uint64_t timestep);

    /// helper function to compute net force/virial
    virtual void computeNetForce(uint64_t timestep);

#ifdef ENABLE_HIP
    /// helper function to compute net force/virial on the GPU
    virtual void computeNetForceGPU(uint64_t timestep);
#endif

#ifdef ENABLE_MPI
    /// helper function to determine the ghost communication flags
    virtual CommFlags determineFlags(uint64_t timestep);

    /// The systems's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    /// Check if any forces introduce anisotropic degrees of freedom
    virtual bool areForcesAnisotropic();
    };

namespace detail
    {
/// Exports the NVEUpdater class to python
void export_Integrator(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
