// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Analyzer.h
    \brief Declares a base class for all analyzers
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __ANALYZER_H__
#define __ANALYZER_H__

#include "Profiler.h"
#include "SystemDefinition.h"
#include "SharedSignal.h"

#include <memory>

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
        Analyzer(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~Analyzer() {};

        //! Abstract method that performs the analysis
        /*! Derived classes will implement this method to calculate their results
            \param timestep Current time step of the simulation
            */
        virtual void analyze(unsigned int timestep){}

        //! Sets the profiler for the analyzer to use
        void setProfiler(std::shared_ptr<Profiler> prof);

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period){}

        //! Print some basic stats to stdout
        /*! Derived classes can optionally implement this function. A System will
            call all of the Analyzers' printStats functions at the end of a run
            so the user can see useful information
        */
        virtual void printStats(){}

        //! Reset stat counters
        /*! If derived classes implement printStats, they should also implement resetStats() to clear any running
            counters printed by printStats. System will reset the stats before any run() so that stats printed
            at the end of the run only apply to that run() alone.
        */
        virtual void resetStats(){}

        //! Get needed pdata flags
        /*! Not all fields in ParticleData are computed by default. When derived classes need one of these optional
            fields, they must return the requested fields in getRequestedPDataFlags().
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            return PDataFlags(0);
            }

        std::shared_ptr<const ExecutionConfiguration> getExecConf()
            {
            return m_exec_conf;
            }

#ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm The Communicator
         */
        virtual void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            m_comm = comm;
            }
#endif
        void addSlot(std::shared_ptr<hoomd::detail::SignalSlot> slot)
            {
            m_slots.push_back(slot);
            }

        void removeDisconnectedSlots()
            {
            for(unsigned int i = 0; i < m_slots.size();)
                {
                if(!m_slots[i]->connected())
                    {
                    m_exec_conf->msg->notice(8) << "Found dead signal @" << std::hex << m_slots[i].get() << std::dec<< std::endl;
                    m_slots.erase(m_slots.begin()+i);
                    }
                else
                    {
                    i++;
                    }
                }
            }
    protected:
        const std::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this analyzer is associated with
        const std::shared_ptr<ParticleData> m_pdata;      //!< The particle data this analyzer is associated with
        std::shared_ptr<Profiler> m_prof;                 //!< The profiler this analyzer is to use

#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;             //!< The communicator to use
#endif

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::vector< std::shared_ptr<hoomd::detail::SignalSlot> > m_slots; //!< Stored shared ptr to the system signals
    };

//! Export the Analyzer class to python
void export_Analyzer(pybind11::module& m);

#endif
