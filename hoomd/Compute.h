// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "SystemDefinition.h"
#include "Profiler.h"
#include "SharedSignal.h"

#include <memory>
#include <string>
#include <vector>

#ifndef __COMPUTE_H__
#define __COMPUTE_H__

/*! \file Compute.h
    \brief Declares a base class for all computes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup computes Computes
    \brief All classes that implement the Compute concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Performs computations on ParticleData structures
/*! The Compute is an abstract concept that performs some kind of computation on the
    particles in a ParticleData structure. This computation is to be done by reading
    the particle data only, no writing. Computes will be used to generate neighbor lists,
    calculate forces, and calculate temperatures, just to name a few.

    For performance and simplicity, each compute is associated with a ParticleData
    on construction. ParticleData pointers are managed with reference counted std::shared_ptr.
    Since each ParticleData cannot change size, this allows the Compute to preallocate
    any data structures that it may need.

    Computes may be referenced more than once and may reference other computes. To prevent
    unneeded data from being calculated, the time step will be passed into the compute
    method so that it can skip calculations if they have already been done this timestep.
    For convenience, the base class will provide a shouldCompute() method that implements
    this behaviour. Derived classes can override if more complicated behavior is needed.

    See \ref page_dev_info for more information
    \ingroup computes
*/
class PYBIND11_EXPORT Compute
    {
    public:
        //! Constructs the compute and associates it with the ParticleData
        Compute(std::shared_ptr<SystemDefinition> sysdef);
        virtual ~Compute() {};

        //! Abstract method that performs the computation
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void compute(unsigned int timestep){}

        //! Abstract method that performs a benchmark
        virtual double benchmark(unsigned int num_iters);

        //! Print some basic stats to stdout
        /*! Derived classes can optionally implement this function. A System will
            call all of the Compute's printStats functions at the end of a run
            so the user can see useful information
        */
        virtual void printStats(){}

        //! Reset stat counters
        /*! If derived classes implement printStats, they should also implement resetStats() to clear any running
            counters printed by printStats. System will reset the stats before any run() so that stats printed
            at the end of the run only apply to that run() alone.
        */
        virtual void resetStats(){}

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


        //! Force recalculation of compute
        /*! If this function is called, recalculation of the compute will be forced (even if had
         *  been calculated earlier in this timestep)
         * \param timestep current timestep
         */
        void forceCompute(unsigned int timestep);

#ifdef ENABLE_MPI
        //! Set communicator this Compute is to use
        /*! \param comm The communicator
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
        const std::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this compute is associated with
        const std::shared_ptr<ParticleData> m_pdata;      //!< The particle data this compute is associated with
        std::shared_ptr<Profiler> m_prof;                 //!< The profiler this compute is to use
#ifdef ENABLE_MPI
        std::shared_ptr<Communicator> m_comm;             //!< The communicator this compute is to use
#endif
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        std::vector< std::shared_ptr<hoomd::detail::SignalSlot> > m_slots; //!< Stored shared ptr to the system signals
        bool m_force_compute;           //!< true if calculation is enforced
        unsigned int m_last_computed;   //!< Stores the last timestep compute was called
        bool m_first_compute;           //!< true if compute has not yet been called

        //! Simple method for testing if the computation should be run or not
        virtual bool shouldCompute(unsigned int timestep);

        //! Peek to see if computation should be run without updating internal state
        virtual bool peekCompute(unsigned int timestep) const;

    private:
        //! The python export needs to be a friend to export shouldCompute()
        friend void export_Compute();
    };

//! Exports the Compute class to python
#ifndef NVCC
void export_Compute(pybind11::module& m);
#endif

#endif
