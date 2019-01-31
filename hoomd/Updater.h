// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "HOOMDMath.h"
#include "SystemDefinition.h"
#include "Profiler.h"
#include "SharedSignal.h"

#include <memory>

#ifndef __UPDATER_H__
#define __UPDATER_H__

/*! \file Updater.h
    \brief Declares a base class for all updaters
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup updaters Updaters
    \brief All classes that implement the Updater concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Performs updates of ParticleData structures
/*! The Updater is an abstract concept that takes a particle data structure and changes it in some way.
    For example, an updater may make a verlet step and update the particle positions to the next timestep.
    Or, it may force a certain particle to be in a certain location. Or, it may sort the particle data
    so that the many Computes suffer far fewer cache misses. The possibilities are endless.

    The base class just defines an update method. Since updaters can reference Compute's, the timestep
    is passed in so that it can be forwarded on to the Compute. Of course, the timestep can also be used
    for time dependent updaters, such as a moving temperature set point. Of course, when an updater is changing
    particle positions/velocities etc... the line between when a timestep begins and ends blurs. See the System class
    for a clear definition.

    See \ref page_dev_info for more information

    \ingroup updaters
*/
class PYBIND11_EXPORT Updater
    {
    public:
        //! Constructs the compute and associates it with the ParticleData
        Updater(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~Updater()  {};

        //! Abstract method that performs the update
        /*! Derived classes will implement this method to perform their specific update
            \param timestep Current time step of the simulation
        */
        virtual void update(unsigned int timestep)  {};

        //! Sets the profiler for the compute to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof);

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

            The base class just returns 0. Derived classes should override this behavior and return
            the calculated value for the given quantity. Only quantities listed in
            the return value getProvidedLogQuantities() will be requested from
            getLogValue().

            See Logger for more information on what this is about.
        */
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            return Scalar(0.0);
            }

        //! Returns a list of log matrix quantities this compute calculates
        /*! The base class implementation just returns an empty vector. Derived classes should override
            this behavior and return a list of quantities that they log.

            See LogMatrix for more information on what this is about.
        */
        virtual std::vector< std::string > getProvidedLogMatrixQuantities()
            {
            return std::vector< std::string >();
            }

        //! Calculates the requested log matrix and returns it
        /*! \param quantity Name of the log quantity to get
            \param timestep Current time step of the simulation

            The base class just returns an empty shared_ptr. Derived classes should override this behavior and return
            the calculated value for the given quantity. Only quantities listed in
            the return value getProvidedLogMatrixQuantities() will be requested from
            getLogMatrixValue().

            See LogMatrix for more information on what this is about.
        */
        virtual pybind11::array getLogMatrix(const std::string& quantity, unsigned int timestep)
            {
            unsigned char tmp[] = {0};
            return pybind11::array(0,tmp);
            }

        //! Print some basic stats to stdout
        /*! Derived classes can optionally implement this function. A System will
            call all of the Updaters' printStats functions at the end of a run
            so the user can see useful information
        */
        virtual void printStats()
            {
            }

        //! Reset stat counters
        /*! If derived classes implement printStats, they should also implement resetStats() to clear any running
            counters printed by printStats. System will reset the stats before any run() so that stats printed
            at the end of the run only apply to that run() alone.
        */
        virtual void resetStats()
            {
            }

        //! Get needed pdata flags
        /*! Not all fields in ParticleData are computed by default. When derived classes need one of these optional
            fields, they must return the requested fields in getRequestedPDataFlags().
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            return PDataFlags(0);
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
        const std::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this compute is associated with
        const std::shared_ptr<ParticleData> m_pdata;      //!< The particle data this compute is associated with
        std::shared_ptr<Profiler> m_prof;                 //!< The profiler this compute is to use
#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;             //!< The communicator this updater is to use
#endif
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::vector< std::shared_ptr<hoomd::detail::SignalSlot> > m_slots; //!< Stored shared ptr to the system signals
    };

//! Export the Updater class to python
void export_Updater(pybind11::module& m);

#endif
