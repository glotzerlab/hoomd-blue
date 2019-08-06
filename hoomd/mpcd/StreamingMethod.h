// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingMethod.h
 * \brief Declaration of mpcd::StreamingMethod
 */

#ifndef MPCD_STREAMING_METHOD_H_
#define MPCD_STREAMING_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/GPUPolymorph.h"
#include "ExternalField.h"
#include "SystemData.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! MPCD streaming method
/*!
 * This method implements the base version of ballistic propagation of MPCD
 * particles.
 */
class PYBIND11_EXPORT StreamingMethod
    {
    public:
        //! Constructor
        StreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                        unsigned int cur_timestep,
                        unsigned int period,
                        int phase);
        //! Destructor
        virtual ~StreamingMethod();

        //! Implementation of the streaming rule
        virtual void stream(unsigned int timestep) { }

        //! Peek if the next step requires streaming
        virtual bool peekStream(unsigned int timestep) const;

        //! Sets the profiler for the integration method to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         *
         * Derived classes should override this to set the parameters of their autotuners.
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            }

        //! Change the timestep
        /*!
         * \param deltaT Fundamental HOOMD integration timestep
         *
         * The streaming step size is set to period * deltaT so that each time
         * the streaming operation is called, the particles advance across the
         * full MPCD interval.
         */
        virtual void setDeltaT(Scalar deltaT)
            {
            m_mpcd_dt = Scalar(m_period)*deltaT;
            }

        //! Get the timestep
        Scalar getDeltaT() const
            {
            return m_mpcd_dt;
            }

        //! Set the external field
        void setField(std::shared_ptr<hoomd::GPUPolymorph<mpcd::ExternalField>> field)
            {
            m_field = field;
            }

        //! Remove the external field
        void removeField()
            {
            m_field.reset();
            }

        //! Set the period of the streaming method
        void setPeriod(unsigned int cur_timestep, unsigned int period);

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;                   //!< MPCD system data
        std::shared_ptr<SystemDefinition> m_sysdef;                     //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_pdata;                        //!< HOOMD particle data
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;               //!< MPCD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;      //!< Execution configuration
        std::shared_ptr<Profiler> m_prof;                               //!< System profiler

        Scalar m_mpcd_dt;               //!< Integration time step
        unsigned int m_period;          //!< Number of MD timesteps between streaming steps
        unsigned int m_next_timestep;   //!< Timestep next streaming step should be performed

        std::shared_ptr<hoomd::GPUPolymorph<mpcd::ExternalField>> m_field;  //!< External field

        //! Check if streaming should occur
        virtual bool shouldStream(unsigned int timestep);
    };

namespace detail
{
//! Export mpcd::StreamingMethod to python
void export_StreamingMethod(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_STREAMING_METHOD_H_
