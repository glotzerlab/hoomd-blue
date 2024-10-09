// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Action.h"
#include "Communicator.h"
#include "HOOMDMath.h"
#include "Trigger.h"

#include <memory>

#ifndef __UPDATER_H__
#define __UPDATER_H__

/*! \file Updater.h
    \brief Declares a base class for all updaters
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup updaters Updaters
    \brief All classes that implement the Updater concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
 */

namespace hoomd
    {
//! Performs updates of ParticleData structures
/*! The Updater is an abstract concept that takes a particle data structure and changes it in some
   way. For example, an updater may make a verlet step and update the particle positions to the next
   timestep. Or, it may force a certain particle to be in a certain location. Or, it may sort the
   particle data so that the many Computes suffer far fewer cache misses. The possibilities are
   endless.

    The base class just defines an update method. Since updaters can reference Compute's, the
   timestep is passed in so that it can be forwarded on to the Compute. Of course, the timestep can
   also be used for time dependent updaters, such as a moving temperature set point. Of course, when
   an updater is changing particle positions/velocities etc... the line between when a timestep
   begins and ends blurs. See the System class for a clear definition.

    See \ref page_dev_info for more information

    \ingroup updaters
*/
class PYBIND11_EXPORT Updater : public Action
    {
    public:
    //! Constructs the compute and associates it with the ParticleData
    Updater(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);
    virtual ~Updater() { };

    //! Abstract method that performs the update
    /*! Derived classes will implement this method to perform their specific update
        \param timestep Current time step of the simulation
    */
    virtual void update(uint64_t timestep) { };

    //! Reset stat counters
    /*! If derived classes provide statistics for the last run, they should resetStats() to
        clear any counters. System will reset the stats before any run() so that stats printed
        at the end of the run only apply to that run() alone.
    */
    virtual void resetStats() { }

    //! Get needed pdata flags
    /*! Not all fields in ParticleData are computed by default. When derived classes need one of
       these optional fields, they must return the requested fields in getRequestedPDataFlags().
    */
    virtual PDataFlags getRequestedPDataFlags()
        {
        return PDataFlags(0);
        }

    std::shared_ptr<const ExecutionConfiguration> getExecConf()
        {
        return m_exec_conf;
        }

    /// Get Trigger
    std::shared_ptr<Trigger> getTrigger()
        {
        return m_trigger;
        }

    /// Set Trigger
    void setTrigger(std::shared_ptr<Trigger> trigger)
        {
        m_trigger = trigger;
        }

    /// Python will notify C++ objects when they are detached from Simulation
    virtual void notifyDetach() { };

    /// Return true if updating should trigger a recount of the degrees of freedom.
    virtual bool mayChangeDegreesOfFreedom(uint64_t timestep)
        {
        return false;
        }

    protected:
    std::shared_ptr<Trigger> m_trigger; /// Trigger that determines if updater runs.
    };

namespace detail
    {
//! Export the Updater class to python
void export_Updater(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
