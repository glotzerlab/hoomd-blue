// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file Analyzer.h
    \brief Declares a base class for all analyzers
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ANALYZER_H__
#define __ANALYZER_H__

#include "Communicator.h"
#include "SharedSignal.h"
#include "SystemDefinition.h"
#include "Trigger.h"

#include <memory>
#include <typeinfo>

namespace hoomd
    {
/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup analyzers Analyzers
    \brief All classes that implement the Analyzer concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
 */

//! Base class for analysis of particle data
/*! An Analyzer is a concept that encapsulates some process that is performed during
    the simulation with the sole purpose of outputting data to the user in some fashion.
    The results of an Analyzer can not modify the simulation in any way, that is what
    the Updater classes are for. In general, analyzers are likely to only be called every 1,000
    time steps or much less often (this value entirely at the user's discretion).
    The System class will handle this. An Analyzer just needs to perform its calculations
    and make its output every time analyze() is called.

    By design Analyzers can reference any number of Computes while performing their
    analysis. The base class provides no methods for doing this, derived classes must
    implement the tracking of the attached Compute classes (via shared pointers)
    themselves. (it is recommended to pass a shared pointer to the Compute
    into the constructor of the derived class).

    See \ref page_dev_info for more information

    \ingroup analyzers
*/
class PYBIND11_EXPORT Analyzer
    {
    public:
    //! Constructs the analyzer and associates it with the ParticleData
    Analyzer(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);
    virtual ~Analyzer() {};

    //! Abstract method that performs the analysis
    /*! Derived classes will implement this method to calculate their results
        \param timestep Current time step of the simulation
        */
    virtual void analyze(uint64_t timestep) { }

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs

        Derived classes should override this to set the parameters of their autotuners.
    */
    virtual void setAutotunerParams(bool enable, unsigned int period) { }

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

    void addSlot(std::shared_ptr<hoomd::detail::SignalSlot> slot)
        {
        m_slots.push_back(slot);
        }

    void removeDisconnectedSlots()
        {
        for (unsigned int i = 0; i < m_slots.size();)
            {
            if (!m_slots[i]->connected())
                {
                m_exec_conf->msg->notice(8) << "Found dead signal @" << std::hex << m_slots[i].get()
                                            << std::dec << std::endl;
                m_slots.erase(m_slots.begin() + i);
                }
            else
                {
                i++;
                }
            }
        }

    /// Python will notify C++ objects when they are detached from Simulation
    virtual void notifyDetach() {};

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

    protected:
    const std::shared_ptr<SystemDefinition>
        m_sysdef; //!< The system definition this analyzer is associated with
    const std::shared_ptr<ParticleData>
        m_pdata; //!< The particle data this analyzer is associated with

    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< Stored shared ptr to the execution configuration
    std::vector<std::shared_ptr<hoomd::detail::SignalSlot>>
        m_slots;                        //!< Stored shared ptr to the system signals
    /// Trigger that determines if updater runs.
    std::shared_ptr<Trigger> m_trigger; 
    };

namespace detail
    {
//! Export the Analyzer class to python
void export_Analyzer(pybind11::module& m);

    } // end namespace detail
    } // end namespace hoomd

#endif
